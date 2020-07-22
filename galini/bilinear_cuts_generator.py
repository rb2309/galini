# Copyright 2020 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implement a class of cuts specific to bilinear expressions."""

import numpy as np

from pyomo.core.expr.calculus.diff_with_pyomo import _diff_SumExpression, _diff_map
from suspect.pyomo.quadratic import QuadraticExpression
import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import differentiate
from coramin.relaxations import relaxation_data_objects
from coramin.relaxations.mccormick import PWMcCormickRelaxation
from coramin.utils.coramin_enums import RelaxationSide


from galini.config import CutsGeneratorOptions, NumericOption
from galini.cuts import CutsGenerator
from galini.math import almost_ge, almost_le
from galini.ipython import embed_ipython


class BilinearCutsGenerator(CutsGenerator):
    name = 'bilinear'

    def __init__(self, galini, config):
        super().__init__(galini, config)
        self.galini = galini
        self.logger = galini.get_logger(__name__)

        self._tolerance = 1e-4

        self._bilinear_terms = None

    @staticmethod
    def cuts_generator_options():
        """Bilinear cuts generator options"""
        return CutsGeneratorOptions(BilinearCutsGenerator.name, [])

    def _detect_bilinear_terms(self, problem, relaxed_problem):
        mc_eps = self.galini.mc.epsilon
        self._bilinear_terms = []
        for relaxation in relaxation_data_objects(relaxed_problem, active=True, descend_into=True):
            if not isinstance(relaxation, PWMcCormickRelaxation):
                continue
            x, y = relaxation.get_rhs_vars()
            z = relaxation.get_aux_var()
            x_lb, x_ub = x.bounds
            y_lb, y_ub = y.bounds
            z_lb, z_ub = z.bounds
            if not np.isclose(x_lb, 0.0, rtol=mc_eps):
                continue
            if not np.isclose(y_lb, 0.0, rtol=mc_eps):
                continue
            if z_lb is None:
                z_lb = -np.inf
            if x_lb is None:
                x_lb = -np.inf
            if y_lb is None:
                y_lb = -np.inf

            if z_ub is None:
                z_ub = np.inf
            if x_ub is None:
                x_ub = np.inf
            if y_ub is None:
                y_ub = np.inf
            lower_gt = z_lb > x_lb * y_lb
            upper_lt = z_ub < x_ub * y_ub
            print('rel')
            print(' -> ', x, x_lb, x_ub)
            print(' -> ', y, y_lb, y_ub)
            print(' -> ', z, z_lb, z_ub)
            print()
            if not (lower_gt or upper_lt):
                continue
            self._bilinear_terms.append((relaxation, x, y, z))

    def before_start_at_root(self, problem, relaxed_problem):
        self._detect_bilinear_terms(problem, relaxed_problem)

    def after_end_at_root(self, problem, relaxed_problem, solution):
        self._bilinear_terms = None

    def before_start_at_node(self, problem, relaxed_problem):
        self._detect_bilinear_terms(problem, relaxed_problem)

    def after_end_at_node(self, problem, relaxed_problem, solution):
        self._bilinear_terms = None

    def has_converged(self, state):
        return not self._bilinear_terms

    def generate(self, problem, relaxed_problem, mip_solution, tree, node):
        cuts = []
        for (rel, x, y, z) in self._bilinear_terms:
            x_lb, x_ub = x.bounds
            y_lb, y_ub = y.bounds
            z_lb, z_ub = z.bounds
            x_val = pe.value(x, exception=False)
            y_val = pe.value(y, exception=False)
            z_val = pe.value(z, exception=False)
            print(x_lb, y_lb, z_lb)
            print(x_ub, y_ub, z_ub)
            if x_val is None or y_val is None or z_val is None:
                continue

            if y_val >= (z_ub / (x_ub * x_ub)) and x_val >= (z_ub / (y_ub * y_ub)) * y_val:
                print('CASE 1')
                if False:
                    z_lb_ub_sq = (np.sqrt(z_lb) + np.sqrt(z_ub))**2.0
                    sqrt_z_lb_ub = np.sqrt(z_lb * z_ub)
                    z_soc = np.sqrt(z_lb_ub_sq * x_val * y_val) - sqrt_z_lb_ub
                    if z_val <= z_soc + self._tolerance:
                        dz_dx = 0.5 * np.sqrt(z_lb_ub_sq * (y_val / x_val))
                        dz_dy = 0.5 * np.sqrt(z_lb_ub_sq * (x_val / y_val))
                        cut_expr = z - dz_dx * x + dz_dx * x_val - dz_dy * y + dz_dy * y_val - z_soc <= 0
                        print(cut_expr)
                        cuts.append(cut_expr)

            if y_val <= (z_ub / (x_ub * x_ub)) * x_val:
                print('CASE 2')

            if x_val <= (z_ub / (y_ub * y_ub)) * y_val:
                print('CASE 3')

            embed_ipython()
        return cuts


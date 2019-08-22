# Copyright 2018 Francesco Ceccon
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

"""Branch & Cut algorithm."""

import numpy as np
from suspect.expression import ExpressionType

from galini.bab.relaxations import ConvexRelaxation
from galini.bab.selection import BestLowerBoundSelectionStrategy
from galini.bab.strategy import KSectionBranchingStrategy
from galini.bab.termination import should_terminate
from galini.core import LinearExpression, SumExpression, Domain
from galini.logging import get_logger
from galini.math import mc, is_close
from galini.relaxations.relaxed_problem import RelaxedProblem
from galini.timelimit import (
    seconds_left,
)

logger = get_logger(__name__)


# pylint: disable=too-many-instance-attributes
class BranchAndCutAlgorithm:
    """Branch and Cut algorithm."""
    name = 'branch_and_cut'

    def __init__(self, galini, solver):
        self.galini = galini
        self.solver = solver
        self._nlp_solver = galini.instantiate_solver('ipopt')
        self._mip_solver = galini.instantiate_solver('mip')
        self._cuts_generators_manager = galini.cuts_generators_manager

        bab_config = galini.get_configuration_group('bab')

        self.tolerance = bab_config['tolerance']
        self.relative_tolerance = bab_config['relative_tolerance']
        self.node_limit = bab_config['node_limit']
        self.fbbt_maxiter = bab_config['fbbt_maxiter']
        self.fbbt_timelimit = bab_config['fbbt_timelimit']
        self.root_node_feasible_solution_seed = \
            bab_config['root_node_feasible_solution_seed']

        self.root_node_feasible_solution_search_timelimit = \
            bab_config['root_node_feasible_solution_search_timelimit']

        bac_config = galini.get_configuration_group('bab.branch_and_cut')
        self.cuts_maxiter = bac_config['maxiter']
        self._use_milp_relaxation = bac_config['use_milp_relaxation']

        self.branching_strategy = KSectionBranchingStrategy(2)
        self.node_selection_strategy = BestLowerBoundSelectionStrategy()

        self._bounds = None
        self._monotonicity = None
        self._convexity = None

    def should_terminate(self, state):
        return should_terminate(
            state, self, paranoid_mode=self.galini.paranoid_mode
        )

    def _cuts_converged(self, state):
        return self._cuts_generators_manager.has_converged(state)

    def _cuts_iterations_exceeded(self, state):
        return state.round > self.cuts_maxiter

    def _perform_fbbt(self, run_id, problem, _tree, node):
        pass


def _convert_linear_expr(linear_problem, expr, objvar=None):
    stack = [expr]
    coefficients = {}
    const = 0.0
    while len(stack) > 0:
        expr = stack.pop()
        if expr.expression_type == ExpressionType.Sum:
            for ch in expr.children:
                stack.append(ch)
        elif expr.expression_type == ExpressionType.Linear:
            const += expr.constant_term
            for ch in expr.children:
                if ch.idx not in coefficients:
                    coefficients[ch.idx] = 0
                coefficients[ch.idx] += expr.coefficient(ch)
        else:
            raise ValueError(
                'Invalid ExpressionType {}'.format(expr.expression_type)
            )

    children = []
    coeffs = []
    for var, coef in coefficients.items():
        children.append(linear_problem.variable(var))
        coeffs.append(coef)

    if objvar is not None:
        children.append(objvar)
        coeffs.append(1.0)

    return LinearExpression(children, coeffs, const)

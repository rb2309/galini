# Copyright 2019 Francesco Ceccon
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

"""Outer-Approximation cuts generator."""
from contextlib import contextmanager

import numpy as np

from galini.config import CutsGeneratorOptions, NumericOption
from galini.core import LinearExpression, Domain
from galini.cuts import CutType, Cut, CutsGenerator
from galini.logging import get_logger
from galini.math import is_close, is_inf, mc, almost_ge, almost_le
from galini.outer_approximation.feasibility_problem import \
    FeasibilityProblemRelaxation
from galini.relaxations.relaxed_problem import RelaxedProblem

logger = get_logger(__name__)


@contextmanager
def fixed_integer_variables(run_id, problem, solution):
    """Context manager to fix integer variables."""
    for i, v in enumerate(problem.variables):
        v_view = problem.variable_view(v)
        if v_view.domain != Domain.REAL:
            value = solution.variables[i].value
            logger.debug(run_id, 'Fix {} to {}', v.name, value)
            v_view.fix(value)

    try:
        yield problem
    finally:
        for v in problem.variables:
            problem.unfix(v)



class OuterApproximationCutsGenerator(CutsGenerator):
    """Implement Outer-Appoximation cuts for Convex MINLP problems.

    References
    ----------
    [0] Bonami P., Biegler L. T., Conn A. R., Cornuéjols G., Grossmann I. E.,
        Laird C. D., Lee J, Lodi A., Margot F., Sawaya N., Wächter A. (2008).
        An algorithmic framework for convex mixed integer nonlinear programs.
        Discrete Optimization
        https://doi.org/10.1016/J.DISOPT.2006.10.011
    [1] Duran, M. A., Grossmann, I. E. (1986).
        An outer-approximation algorithm for a class of mixed-integer nonlinear
        programs.
        Mathematical Programming
        https://doi.org/10.1007/BF02592064
    """

    name = 'outer_approximation'

    def __init__(self, galini, config):
        super().__init__(galini, config)
        self.galini = galini

        self._round = 0
        self._feasibility_problem = None
        self._nlp_solution = None

        self._nlp_solver = galini.instantiate_solver('ipopt')

        self.convergence_relative_tol = config['convergence_relative_tol']
        self.threshold = config['threshold']

    def _reset_node_local_storage(self):
        self._round = 0
        self._feasibility_problem = None
        self._nlp_solution = None

    @staticmethod
    def cuts_generator_options():
        """Outer Approximation cuts generator options"""
        return CutsGeneratorOptions(OuterApproximationCutsGenerator.name, [
            NumericOption('threshold', default=1e-3),
            NumericOption('convergence_relative_tol', default=1e-3),
        ])

    def before_start_at_root(self, run_id, problem, relaxed_problem):
        self._reset_node_local_storage()

    def after_end_at_root(self, run_id, problem, relaxed_problem, solution):
        pass

    def before_start_at_node(self, run_id, problem, relaxed_problem):
        self._reset_node_local_storage()

    def after_end_at_node(self, run_id, problem, relaxed_problem, solution):
        pass

    def has_converged(self, state):
        if self._nlp_solution is None:
            return False

        if is_inf(state.lower_bound):
            return False

        return is_close(
            state.lower_bound,
            self._nlp_solution,
            rtol=self.convergence_relative_tol,
        )

    def generate(self, run_id, problem, relaxed_problem, linear_problem,
                 mip_solution, tree, node):
        logger.debug(run_id, 'Starting cut round={}', self._round)

        noninteger_values = self._compute_noninteger_values(
            run_id, relaxed_problem, mip_solution
        )

        if noninteger_values is None:
            return []

        cuts_gen = self._generate_cuts(
            run_id,
            relaxed_problem,
            linear_problem,
            mip_solution,
            noninteger_values,
        )
        cuts = list(cuts_gen)
        self._round += 1
        return cuts

    def _generate_cuts(self, run_id, relaxed_problem, linear_problem,
                       mip_solution, x_k):
        variables = relaxed_problem.variables

        nonlinear_objective = self._nonlinear_objective(relaxed_problem)
        if nonlinear_objective:
            raise RuntimeError('Relaxation has non linear objective.')

        # Filter out constraints that are not of interest
        nonlinear_constraints = self._nonlinear_constraints(relaxed_problem)

        # What if it's a nonlinear constraint non present in the original
        # problem?
        linear_constraints = [
            linear_problem.constraint(c.name)
            for c in nonlinear_constraints
        ]

        logger.debug(
            run_id,
            'Number of Nonlinear constraints: {}',
            len(nonlinear_constraints),
        )

        linear_expr_idx = [c.root_expr.idx for c in linear_constraints]
        nonlinear_expr_idx = [c.root_expr.idx for c in nonlinear_constraints]

        mip_values = [v.value for v in mip_solution.variables]
        nonlinear_mip_values = mip_values[:relaxed_problem.num_variables]

        linear_expr_eval = \
            linear_problem.expression_tree_data()\
                .eval(mip_values, linear_expr_idx)

        nonlinear_expr_eval = \
            relaxed_problem.expression_tree_data()\
                .eval(nonlinear_mip_values, nonlinear_expr_idx)

        linear_expr_value = linear_expr_eval.forward(0, mip_values)
        linear_expr_value = np.array(linear_expr_value)

        nonlinear_expr_value = \
            nonlinear_expr_eval.forward(0, nonlinear_mip_values)
        nonlinear_expr_value = np.array(nonlinear_expr_value)

        # The difference between the linear cut and the convex function
        # evaluated at the mip solution
        nonlinear_linear_diff = nonlinear_expr_value - linear_expr_value

        logger.debug(run_id, 'Nonlinear - linear = {}', nonlinear_linear_diff)

        w = np.zeros_like(nonlinear_constraints)
        nonlinear_expr_x_k_value = nonlinear_expr_eval.forward(0, x_k)

        for i, constraint in enumerate(nonlinear_constraints):
            new_cut = self._generate_cut_for_constraint(
                i,
                constraint,
                variables,
                x_k,
                w,
                nonlinear_linear_diff,
                nonlinear_expr_eval,
                nonlinear_expr_x_k_value,
            )

            if new_cut is not None:
                yield new_cut

    def _generate_cut_for_constraint(self, i, constraint, variables, x_k, w,
                                     nonlinear_linear_diff, nonlinear_expr_eval,
                                     nonlinear_expr_x_k_value):
        cons_lb = constraint.lower_bound
        cons_ub = constraint.upper_bound

        if cons_lb is None:
            # g(x) <= c, g(x) convex
            # If nonlinear linear diff is >= threshold, then
            # f(x) - L(x) > threshold
            above_threshold = almost_ge(
                nonlinear_linear_diff[i],
                self.threshold,
                atol=mc.epsilon,
            )
            if above_threshold:
                return self._generate_cut(
                    i, constraint, variables, x_k, w,
                    nonlinear_expr_eval, nonlinear_expr_x_k_value
                )
        elif cons_ub is None:
            # -g(x) >= c, g(x) convex
            # This is the same as before, but the (convex) constraint is
            # written as -g(x) >= c, where g(x) is convex
            below_threshold = almost_le(
                nonlinear_linear_diff[i],
                -self.threshold,
                atol=mc.epsilon,
            )
            if below_threshold:
                return self._generate_cut(
                    i, constraint, variables, x_k, w,
                    nonlinear_expr_eval, nonlinear_expr_x_k_value
                )
        else:
            # c <= g(x) <= c, g(x) convex
            above_threshold = almost_ge(
                nonlinear_linear_diff[i],
                self.threshold,
                atol=mc.epsilon,
            )
            if above_threshold:
                return self._generate_cut(
                    i, constraint, variables, x_k, w,
                    nonlinear_expr_eval, nonlinear_expr_x_k_value
                )

        return None


    def _nonlinear_constraints(self, relaxed_problem):
        if self.galini.paranoid_mode:
            assert all(
                g.root_expr.polynomial_degree() >= 0
                for g in relaxed_problem.constraints
            )

        return [
            constraint
            for constraint in relaxed_problem.constraints
            if constraint.root_expr.polynomial_degree() > 1
        ]

    def _nonlinear_objective(self, relaxed_problem):
        if self.galini.paranoid_mode:
            assert relaxed_problem.objective.root_expr.polynomial_degree() >= 0

        if relaxed_problem.objective.root_expr.polynomial_degree() > 1:
            return relaxed_problem.objective

        return None

    def _compute_noninteger_values(self, run_id, relaxed_problem, mip_solution):
        logger.debug(run_id, 'Compute points for non integer variables')

        f_x_solution = self._solve_nlp_with_integer_fixed(
            run_id, relaxed_problem, mip_solution
        )

        logger.debug(run_id, 'Solving NLP returned {}', f_x_solution)

        if not f_x_solution:
            logger.warning(run_id, 'No solution from NLP. Generating no cuts')
            return None

        assert f_x_solution.status.is_success()
        self._nlp_solution = f_x_solution.objective_value()

        x_k = [
            v.value
            for v in f_x_solution.variables[:relaxed_problem.num_variables]
        ]
        return x_k

    def _solve_nlp_with_integer_fixed(self, run_id, relaxed_problem,
                                      mip_solution):

        with fixed_integer_variables(run_id,
                                     relaxed_problem,
                                     mip_solution) as problem:

            nlp_solution = self._nlp_solver.solve(problem)

            logger.debug(run_id, 'NLP with integer fixed: {}', nlp_solution)

        if nlp_solution.status.is_success():
            return nlp_solution

        feasibility_nlp_solution = self._solve_feasibility_problem(
            run_id, relaxed_problem, mip_solution
        )

        return feasibility_nlp_solution

    def _solve_feasibility_problem(self, run_id, problem, mip_solution):
        if self._feasibility_problem is None:
            # Lazily create feasibility problem and save it for next time
            relaxation = FeasibilityProblemRelaxation()
            self._feasibility_problem = RelaxedProblem(relaxation, problem)

        with fixed_integer_variables(run_id,
                                     self._feasibility_problem.relaxed,
                                     mip_solution) as nlp_problem:

            feasibility_nlp_solution = self._nlp_solver.solve(nlp_problem)

            logger.debug(
                run_id,
                'Feasibility NLP with integer fixed: {}',
                feasibility_nlp_solution
            )

            if feasibility_nlp_solution.status.is_success():
                return feasibility_nlp_solution

            logger.warning(
                run_id,
                'Feasibility NLP returned a non feasible solution'
            )
            return None

    def _generate_cut(self, i, constraint, x, x_k, w, fg, g_x):
        w[i] = 1.0
        d_fg = fg.reverse(1, w)
        w[i] = 0.0

        cut_name = self._cut_name(i, constraint.name)
        cut_expr = LinearExpression(x, d_fg, -np.dot(d_fg, x_k) + g_x[i])

        return Cut(
            CutType.LOCAL,
            cut_name,
            cut_expr,
            constraint.lower_bound,
            constraint.upper_bound,
            is_objective=False,
        )

    def _cut_name(self, i, name):
        return '_outer_approximation_{}_{}_r_{}'.format(i, name, self._round)

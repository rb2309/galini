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

"""Branch & Cut algorithm."""
import numpy as np
from collections import namedtuple
from galini.bab.relaxations import ConvexRelaxation, LinearRelaxation
from galini.bab.termination import has_timeout
from galini.bab.node import NodeSolution
from galini.core import LinearExpression, SumExpression, Domain
from galini.math import is_close, mc
from galini.util import expr_to_str
from galini.relaxations.relaxed_problem import RelaxedProblem
from galini.logging import get_logger, DEBUG
from galini.branch_and_cut.cuts import CutsState
from galini.branch_and_cut.primal import (
    solve_convex_problem,
    find_root_node_feasible_solution,
    solve_primal,
    solve_primal_with_solution,
)
from galini.branch_and_cut.bounds import (
    perform_fbbt_on_problem,
    tighten_bounds_on_problem,
)
from galini.branch_and_cut.termination import cut_loop_should_terminate


logger = get_logger(__name__)


ProblemStructure = namedtuple(
    'ProblemStructure',
    ['bounds', 'monotonicity', 'convexity'],
)


class BranchAndCutAlgorithm:
    def __init__(self, galini):
        self._cuts_generators_manager = galini.cuts_generators_manager
        self._mip_solver = galini.instantiate_solver('mip')
        self._nlp_solver = galini.instantiate_solver('ipopt')

    def solve_problem_at_root(self, run_id, problem, tree, node, config):
        # Perform FBBT and update bounds
        bounds, monotonicity, convexity = perform_fbbt_on_problem(
            run_id, problem, config.fbbt_maxiter, config.fbbt_timelimit
        )

        structure = ProblemStructure(bounds, monotonicity, convexity)

        tighten_bounds_on_problem(run_id, problem, bounds)

        # At root node we also need to build the convex relaxation that will
        # be used at other nodes.
        relaxation = ConvexRelaxation(
            problem,
            bounds,
            monotonicity,
            convexity,
        )
        relaxed = RelaxedProblem(relaxation, problem)
        convex_problem = relaxed.relaxed

        self._cuts_generators_manager.before_start_at_root(
            run_id, problem, convex_problem
        )

        # Do solve problem with cuts loop
        solution = self._solve_problem_at_node(
            run_id, problem, convex_problem, tree, node, config, structure,
        )

        self._cuts_generators_manager.after_end_at_root(
            run_id, problem, convex_problem, solution,
        )

        return solution, convex_problem

    def solve_problem_at_node(self, run_id, problem, convex_problem, tree,
                              node, config):

        # Perform FBBT and update bounds
        bounds, monotonicity, convexity = perform_fbbt_on_problem(
            run_id, problem, config.fbbt_maxiter, config.fbbt_timelimit
        )

        structure = ProblemStructure(bounds, monotonicity, convexity)

        tighten_bounds_on_problem(run_id, problem, bounds)

        # At root node we also need to build the convex relaxation that will
        # be used at other nodes.
        relaxation = ConvexRelaxation(
            problem,
            bounds,
            monotonicity,
            convexity,
        )
        relaxed = RelaxedProblem(relaxation, problem)
        convex_problem = relaxed.relaxed

        self._cuts_generators_manager.before_start_at_node(
            run_id, problem, convex_problem
        )

        # Do solve problem with cuts loop
        solution = self._solve_problem_at_node(
            run_id, problem, convex_problem, tree, node, config, structure
        )

        self._cuts_generators_manager.after_end_at_node(
            run_id, problem, convex_problem, solution,
        )

        return solution

    def _solve_problem_at_node(self, run_id, problem, convex_problem,
                               tree, node, config, structure):
        logger.info(
            run_id,
            'Starting Cut generation iterations. Maximum iterations={}',
            config.cuts.maxiter)

        generators_name = [
            g.name for g in self._cuts_generators_manager.generators
        ]

        logger.info(
            run_id,
            'Using cuts generators: {}',
            ', '.join(generators_name)
        )

        if logger.level <= DEBUG:
            logger.debug(run_id, 'Relaxed Convex Problem')
            logger.debug(run_id, 'Variables:')
            relaxed = convex_problem

            for v in relaxed.variables:
                vv = relaxed.variable_view(v)
                logger.debug(
                    run_id, '\t{}: [{}, {}] c {}',
                    v.name, vv.lower_bound(), vv.upper_bound(), vv.domain
                )
            logger.debug(
                run_id, 'Objective: {}',
                expr_to_str(relaxed.objective.root_expr)
            )
            logger.debug(run_id, 'Constraints:')
            for constraint in relaxed.constraints:
                logger.debug(
                    run_id,
                    '{}: {} <= {} <= {}',
                    constraint.name,
                    constraint.lower_bound,
                    expr_to_str(constraint.root_expr),
                    constraint.upper_bound,
                )

        # Check if problem is convex in current domain, in that case
        # use IPOPT to solve it (if all variables are reals)
        if structure and _is_convex(problem, structure.convexity):
            all_reals = all(
                problem.variable_view(v).domain.is_real()
                for v in problem.variables
            )
            if all_reals:
                return solve_convex_problem(problem, self._nlp_solver)

        if not node.has_parent:
            # It's root node, try to find a feasible integer solution
            feasible_solution = find_root_node_feasible_solution(
                run_id, problem, config, self._nlp_solver
            )
            logger.info(
                run_id, 'Initial feasible solution: {}', feasible_solution
            )
        else:
            feasible_solution = None

        linear_problem = \
            self._build_linear_relaxation(convex_problem, structure)

        cuts_state = CutsState()

        mip_solution = None

        originally_integer = []
        if not config.cuts.use_milp_relaxation:
            for var in linear_problem.relaxed.variables:
                vv = linear_problem.relaxed.variable_view(var)
                if vv.domain.is_integer():
                    originally_integer.append(var)
                    linear_problem.relaxed.set_domain(var, Domain.REAL)

        while (not self._cuts_converged(cuts_state) and
               not has_timeout() and
               not cut_loop_should_terminate(cuts_state, config.cuts)):

            feasible, new_cuts, mip_solution = self._perform_cut_round(
                run_id, problem, convex_problem,
                linear_problem.relaxed, cuts_state, tree, node
            )

            if not feasible:
                return NodeSolution(mip_solution, feasible_solution)

            # Add cuts as constraints
            # TODO(fra): use problem global and local cuts
            for cut in new_cuts:
                if not cut.is_objective:
                    linear_problem.add_constraint(
                        cut.name,
                        cut.expr,
                        cut.lower_bound,
                        cut.upper_bound,
                    )
                else:
                    objvar = linear_problem.relaxed.variable('_objvar')
                    assert cut.lower_bound is None
                    assert cut.upper_bound is None
                    new_root_expr = SumExpression([
                        cut.expr,
                        LinearExpression([objvar], [-1.0], 0.0)
                    ])
                    linear_problem.add_constraint(
                        cut.name,
                        new_root_expr,
                        None,
                        0.0
                    )

            logger.debug(
                run_id, 'Updating CutState: State={}, Solution={}',
                cuts_state, mip_solution
            )

            cuts_state.update(
                mip_solution,
                paranoid=config.paranoid_mode,
                atol=config.termination.tolerance,
                rtol=config.termination.relative_tolerance,
            )

            if not new_cuts:
                break

        logger.debug(
            run_id,
            'Lower Bound from MIP = {}; Tree Upper Bound = {}',
            cuts_state.lower_bound,
            tree.upper_bound
        )

        if not config.cuts.use_milp_relaxation:
            for var in originally_integer:
                linear_problem.relaxed.set_domain(var, Domain.INTEGER)

            # Solve MILP to obtain MILP solution
            mip_solution = self._mip_solver.solve(linear_problem.relaxed)

        if cuts_state.lower_bound >= tree.upper_bound and \
                not is_close(cuts_state.lower_bound, tree.upper_bound,
                             atol=mc.epsilon):
            # No improvement
            return NodeSolution(mip_solution, None)

        if has_timeout():
            # No time for finding primal solution
            return NodeSolution(mip_solution, None)

        primal_solution = solve_primal_with_solution(
            problem, mip_solution, self._nlp_solver, fix_all=True
        )
        new_primal_solution = \
            solve_primal(problem, mip_solution, self._nlp_solver)

        if new_primal_solution is not None:
            primal_solution = new_primal_solution

        if not primal_solution.status.is_success() and \
                feasible_solution is not None:
            # Could not get primal solution, but have a feasible solution
            return NodeSolution(mip_solution, feasible_solution)

        return NodeSolution(mip_solution, primal_solution)

    def _cuts_converged(self, state):
        return self._cuts_generators_manager.has_converged(state)

    def _build_linear_relaxation(self, problem, structure):
        relaxation = LinearRelaxation(
            problem,
            structure.bounds,
            structure.monotonicity,
            structure.convexity,
        )
        return RelaxedProblem(relaxation, problem)

    def _perform_cut_round(self, run_id, problem, relaxed_problem,
                           linear_problem, cuts_state, tree, node):

        logger.debug(
            run_id, 'Round {}. Solving linearized problem.', cuts_state.round
        )

        mip_solution = self._mip_solver.solve(linear_problem)

        logger.debug(
            run_id,
            'Round {}. Linearized problem solution is {}',
            cuts_state.round, mip_solution.status.description()
        )
        logger.debug(run_id, 'Objective is {}'.format(mip_solution.objective))
        logger.debug(run_id, 'Variables are {}'.format(mip_solution.variables))

        if not mip_solution.status.is_success():
            return False, None, mip_solution

        # Generate new cuts
        new_cuts = self._cuts_generators_manager.generate(
            run_id, problem, relaxed_problem, linear_problem, mip_solution,
            tree, node
        )
        logger.debug(
            run_id, 'Round {}. Adding {} cuts.',
            cuts_state.round, len(new_cuts)
        )
        return True, new_cuts, mip_solution


def _is_convex(problem, cvx_map):
    obj = problem.objective
    is_objective_cvx = cvx_map[obj.root_expr].is_convex()

    if not is_objective_cvx:
        return False

    return all(
        _constraint_is_convex(cvx_map, cons)
        for cons in problem.constraints
    )


def _constraint_is_convex(cvx_map, cons):
    cvx = cvx_map[cons.root_expr]
    # g(x) <= UB
    if cons.lower_bound is None:
        return cvx.is_convex()

    # g(x) >= LB
    if cons.upper_bound is None:
        return cvx.is_concave()

    # LB <= g(x) <= UB
    return cvx.is_linear()

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

"""Branch and Cut Solver."""


import numpy as np

from galini.bab.selection import BestLowerBoundSelectionStrategy
from galini.bab.solution import BabSolution, BabStatusInterrupted
from galini.bab.strategy import KSectionBranchingStrategy
from galini.bab.telemetry import BranchAndCutTelemetry
from galini.bab.termination import should_terminate
from galini.bab.tree import BabTree
from galini.branch_and_cut.algorithm import BranchAndCutAlgorithm
from galini.branch_and_cut.bounds import (
    perform_obbt_on_model,
)
from galini.branch_and_cut.configuration import BranchAndCutConfiguration
from galini.config import (
    SolverOptions,
    NumericOption,
    IntegerOption,
    BoolOption,
    OptionsGroup,
)
from galini.logging import get_logger
from galini.math import is_close
from galini.solvers import Solver, OptimalObjective, OptimalVariable

logger = get_logger(__name__)


class BranchAndCutSolver(Solver):
    name = 'bac'

    description = 'Generic Branch & Cut solver.'

    def __init__(self, galini):
        super().__init__(galini)

        self.bac_config = BranchAndCutConfiguration(
            galini.get_configuration_group('bac'),
            paranoid_mode=galini.paranoid_mode,
        )

        self._telemetry = BranchAndCutTelemetry(galini.telemetry)

        self._tree = None
        self._termination_state = None

    @staticmethod
    def solver_options():
        return SolverOptions(BranchAndCutSolver.name, [
            NumericOption('tolerance', default=1e-6),
            NumericOption('relative_tolerance', default=1e-6),
            IntegerOption('node_limit', default=100000000),
            IntegerOption('root_node_feasible_solution_seed', default=None),
            NumericOption(
                'root_node_feasible_solution_search_timelimit',
                default=6000000,
            ),
            IntegerOption('fbbt_maxiter', default=10),
            IntegerOption('obbt_simplex_maxiter', default=1000),
            NumericOption('obbt_timelimit', default=6000000),
            NumericOption('fbbt_timelimit', default=6000000),
            IntegerOption('fbbt_max_quadratic_size', default=1000),
            IntegerOption('fbbt_max_expr_children', default=1000),
            BoolOption('catch_keyboard_interrupt', default=True),
            OptionsGroup('branch_and_cut', [
                IntegerOption(
                    'maxiter',
                    default=20,
                    description='Number of cut rounds'
                ),
                BoolOption(
                    'use_milp_relaxation',
                    default=False,
                    description='Solve MILP relaxations, not LP'
                )
            ]),
        ])

    def before_solve(self, model, problem):
        try:
            perform_obbt_on_model(
                model,
                problem,
                self.bac_config.obbt_timelimit,
                self.bac_config.obbt_simplex_maxiter,
            )
        except TimeoutError:
            logger.info(0, 'OBBT timed out')
            return

        except Exception as ex:
            logger.warning(0, 'Error performing OBBT: {}', ex)
            raise

    def get_branching_strategy(self):
        """Get the branching strategy."""
        return KSectionBranchingStrategy(2)

    def get_node_selection_strategy(self):
        """Get the node selection strategy."""
        return BestLowerBoundSelectionStrategy()

    def actual_solve(self, problem, run_id, **kwargs):
        # Run bab loop, catch keyboard interrupt from users
        keyboard_interrupt = False
        if self.bac_config.catch_keyboard_interrupt:
            try:
                self._bab_loop(problem, run_id, **kwargs)
            except KeyboardInterrupt:
                keyboard_interrupt = True
        else:
            self._bab_loop(problem, run_id, **kwargs)
        assert self._tree is not None
        termination_reason = should_terminate(
            self._tree.state,
            self.bac_config.termination,
        )
        return _solution_from_tree(
            problem, self._tree, termination_reason
        )

    def _bab_loop(self, problem, run_id, **kwargs):
        # Setup of Bab objects
        branching_strategy = self.get_branching_strategy()
        node_selection_strategy = self.get_node_selection_strategy()
        algo = BranchAndCutAlgorithm(self.galini)
        tree = BabTree(problem, branching_strategy, node_selection_strategy)
        self._tree = tree

        # Solve root problem and build convex relaxation
        logger.info(run_id, 'Solving root problem')
        root_solution, convex_problem = algo.solve_problem_at_root(
            run_id, problem, tree, tree.root, self.bac_config
        )
        tree.update_root(root_solution)
        self._telemetry.end_iteration(run_id, tree)

        logger.info(run_id, 'Root problem solved, tree state {}', tree.state)
        logger.log_add_bab_node(
            run_id,
            coordinate=[0],
            lower_bound=tree.root.lower_bound,
            upper_bound=tree.root.upper_bound,
        )

        # Enter branch and bound loop
        while not should_terminate(tree.state, self.bac_config.termination,
                                   paranoid_mode=self.galini.paranoid_mode):
            logger.info(
                run_id,
                'Tree state at beginning of iteration: {}',
                tree.state,
            )

            # No more nodes to visit, terminate
            if not tree.has_nodes():
                logger.info(run_id, 'No more nodes to visit.')
                break

            # Continue branch and bound
            current_node = tree.next_node()
            if current_node.parent is None:
                # This is the root node.
                node_children, branching_point = \
                    tree.branch_at_node(current_node)
                logger.info(run_id, 'Branched at point {}', branching_point)
                continue
            else:
                var_view = \
                    current_node.problem.variable_view(current_node.variable)

                branching_variable = (
                    current_node.variable.name,
                    var_view.lower_bound(),
                    var_view.upper_bound(),
                )
                logger.log_add_bab_node(
                    run_id,
                    coordinate=current_node.coordinate,
                    lower_bound=current_node.parent.lower_bound,
                    upper_bound=current_node.parent.upper_bound,
                    branching_variables=[branching_variable],
                )

            logger.info(
                run_id,
                'Visiting node {}: parent state={}, parent solution={}',
                current_node.coordinate,
                current_node.parent.state,
                current_node.parent.state.upper_bound_solution,
            )

            if current_node.parent.lower_bound >= tree.upper_bound:
                logger.info(
                    run_id,
                    "Phatom node because it won't improve bound: node.lower_bound={}, tree.upper_bound={}",
                    current_node.parent.lower_bound,
                    tree.upper_bound,
                )
                logger.log_prune_bab_node(run_id, current_node.coordinate)
                tree.phatom_node(current_node)
                self._telemetry.end_iteration(run_id, tree)
                continue

            # Solve problem at node
            solution = algo.solve_problem_at_node(
                run_id, problem, convex_problem, tree, current_node,
                self.bac_config,
            )
            tree.update_node(current_node, solution)

            current_node_converged = is_close(
                solution.lower_bound,
                solution.upper_bound,
                atol=self.bac_config.termination.tolerance,
                rtol=self.bac_config.termination.relative_tolerance,
            )

            if not current_node_converged:
                node_children, branching_point = tree.branch_at_node(current_node)
                logger.info(run_id, 'Branched at point {}', branching_point)
            else:
                # We won't explore this part of the tree anymore.
                # Add to phatomed nodes.
                logger.info(run_id, 'Phatom node {}', current_node.coordinate)
                logger.log_prune_bab_node(run_id, current_node.coordinate)
                tree.phatom_node(current_node)

            _log_problem_information_at_node(
                run_id, current_node.problem, solution, current_node)
            logger.info(run_id, 'New tree state at {}: {}', current_node.coordinate, tree.state)
            logger.update_variable(run_id, 'z_l', tree.nodes_visited, tree.lower_bound)
            logger.update_variable(run_id, 'z_u', tree.nodes_visited, tree.upper_bound)
            logger.info(
                run_id,
                'Child {} has solutions: LB={} UB={}',
                current_node.coordinate,
                solution.lower_bound_solution,
                solution.upper_bound_solution,
            )
            self._telemetry.end_iteration(run_id, tree)

        termination_reason = should_terminate(
            tree.state, self.bac_config.termination,
            paranoid_mode=self.galini.paranoid_mode
        )

        logger.info(run_id, 'Branch & Bound Finished: {}', tree.state)
        logger.info(
            run_id, 'Branch & Bound Termination: {}', termination_reason
        )


def _solution_from_tree(problem, tree, termination_reason):
    nodes_visited = tree.nodes_visited

    if len(tree.solution_pool) == 0:
        # Return lower bound only
        optimal_obj = [
            OptimalObjective(name=problem.objective.name, value=None)
        ]
        optimal_vars = [
            OptimalVariable(name=v.name, value=None)
            for v in problem.variables
        ]
        return BabSolution(
            BabStatusInterrupted(),
            optimal_obj,
            optimal_vars,
            dual_bound=tree.state.lower_bound,
            nodes_visited=nodes_visited,
        )

    primal_solution = tree.solution_pool.head

    if termination_reason is not None:
        is_timeout = termination_reason.is_timeout()
        has_converged = termination_reason.has_converged()
        node_limit_exceeded = termination_reason.is_iterations_exceeded()
    else:
        is_timeout = has_converged = node_limit_exceeded = False

    return BabSolution(
        primal_solution.status,
        primal_solution.objectives,
        primal_solution.variables,
        dual_bound=tree.state.lower_bound,
        nodes_visited=nodes_visited,
        nodes_remaining=len(tree.open_nodes),
        is_timeout=is_timeout,
        has_converged=has_converged,
        node_limit_exceeded=node_limit_exceeded,
    )


def _log_problem_information_at_node(run_id, problem, solution, node):
    group_name = '_'.join([str(c) for c in node.coordinate])
    logger.tensor(
        run_id,
        group=group_name,
        dataset='lower_bounds',
        data=np.array(problem.lower_bounds)
    )
    logger.tensor(
        run_id,
        group=group_name,
        dataset='upper_bounds',
        data=np.array(problem.upper_bounds)
    )
    solution = solution.upper_bound_solution

    if solution is None:
        return

    if solution.status.is_success():
        logger.tensor(
            run_id,
            group=group_name,
            dataset='solution',
            data=np.array([v.value for v in solution.variables]),
        )

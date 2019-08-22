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

"""Branch & Cut Primal finding."""

import numpy as np
import datetime
from galini.core import Domain
from galini.timelimit import current_time, seconds_left
from galini.logging import get_logger
from galini.math import is_close, mc
from galini.bab.node import NodeSolution
from galini.bab.termination import has_timeout

logger = get_logger(__name__)


def solve_convex_problem(problem, nlp_solver):
    """Solve a convex continuous problem."""
    solution = nlp_solver.solve(problem)
    return NodeSolution(solution, solution)


def find_root_node_feasible_solution(run_id, problem, config, nlp_solver):
    """Find feasible solution at root node."""
    logger.info(run_id, 'Finding feasible solution at root node')

    if config.root_node_feasible_solution_seed is not None:
        seed = config.root_node_feasible_solution_seed
        logger.info(run_id, 'Use numpy seed {}', seed)
        np.random.seed(seed)

    if not problem.has_integer_variables():
        return _find_root_node_feasible_solution_continuous(
            run_id, problem, config, nlp_solver,
        )

    return _find_root_node_feasible_solution_mixed_integer(
        run_id, problem, config, nlp_solver,
    )


def _find_root_node_feasible_solution_continuous(_run_id, problem, config,
                                                 nlp_solver):
    start_time = current_time()
    feasible_solution_search_time = min(
        datetime.timedelta(
            seconds=config.root_node_feasible_solution_search_timelimit
        ),
        datetime.timedelta(seconds=seconds_left())
    )
    end_time = start_time + feasible_solution_search_time

    # Can't pass 0 as time limit to ipopt
    now = current_time()
    if end_time <= start_time:
        return None
    time_left = max(1, (end_time - now).seconds)
    return nlp_solver.solve(problem, timelimit=time_left)


def _find_root_node_feasible_solution_mixed_integer(run_id, problem,
                                                    config, nlp_solver):
    feasible_solution = None
    is_timeout = False
    start_time = current_time()
    feasible_solution_search_time = min(
        datetime.timedelta(
            seconds=config.root_node_feasible_solution_search_timelimit
        ),
        datetime.timedelta(seconds=seconds_left())
    )
    end_time = start_time + feasible_solution_search_time
    iteration = 1

    if end_time <= start_time:
        return None

    while not feasible_solution and not is_timeout:
        if has_timeout():
            break

        for v in problem.variables:
            vv = problem.variable_view(v)
            if not vv.domain.is_real():
                # check if it has starting point
                lb = vv.lower_bound()
                ub = vv.upper_bound()
                is_integer = vv.domain.is_integer()
                if is_close(lb, ub, atol=mc.epsilon):
                    fixed_point = lb
                else:
                    if is_integer:
                        lb = min(lb, -mc.integer_infinity)
                        ub = min(ub + 1, mc.integer_infinity)
                    fixed_point = np.random.randint(lb, ub)
                vv.fix(fixed_point)

        now = current_time()
        if now > end_time or has_timeout():
            is_timeout = True
        else:
            # Can't pass 0 as time limit to ipopt
            time_left = max(1, (end_time - now).seconds)
            solution = nlp_solver.solve(problem, timelimit=time_left)
            if solution.status.is_success():
                feasible_solution = solution

            logger.info(
                run_id, 'Iteration {}: Solution is {}',
                iteration, solution.status.description()
            )

        iteration += 1

    # unfix all variables
    for v in problem.variables:
        problem.unfix(v)

    return feasible_solution


def solve_primal(problem, mip_solution, nlp_solver):
    solution = solve_primal_with_solution(problem, mip_solution, nlp_solver)
    if solution.status.is_success():
        return solution

    # Try solutions from mip solution pool, if available
    if mip_solution.solution_pool is None:
        return solution

    for mip_solution_from_pool in mip_solution.solution_pool:
        if seconds_left() <= 0:
            return solution

        solution_from_pool = solve_primal_with_solution(
            problem, mip_solution_from_pool.inner, nlp_solver
        )
        if solution_from_pool.status.is_success():
            return solution_from_pool

    # No solution from pool was feasible, return original infeasible sol
    return solution


def solve_primal_with_solution(problem, mip_solution, nlp_solver,
                               fix_all=False):
    # Solve original problem
    # Use mip solution as starting point
    for v, sv in zip(problem.variables, mip_solution.variables):
        domain = problem.domain(v)
        view = problem.variable_view(v)
        if sv.value is None:
            lb = view.lower_bound()
            if lb is None:
                lb = -mc.infinity
            ub = view.upper_bound()
            if ub is None:
                ub = mc.infinity

            value = lb + (ub - lb) / 2.0
        else:
            value = sv.value

        if domain != Domain.REAL:
            # Solution (from pool) can contain non integer values for
            # integer variables. Simply round these values up
            if not is_close(np.trunc(value), value, atol=mc.epsilon):
                value = min(view.upper_bound(), np.ceil(value))
            problem.fix(v, value)
        elif fix_all:
            problem.fix(v, value)
        else:
            problem.set_starting_point(v, value)

    solution = nlp_solver.solve(problem)

    # unfix all variables
    for v in problem.variables:
        problem.unfix(v)

    return solution

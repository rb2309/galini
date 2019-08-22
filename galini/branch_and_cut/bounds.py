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

"""Perform FBBT on problem"""

import numpy as np
import pyomo.environ as pe
import coramin.domain_reduction.obbt as coramin_obbt
from coramin.relaxations.auto_relax import relax
from pyomo.core.expr.current import identify_variables
from pyomo.core.kernel.component_set import ComponentSet
from suspect.interval import Interval
from galini.logging import get_logger
from galini.special_structure import (
    propagate_special_structure,
    perform_fbbt,
)
from galini.timelimit import (
    current_time,
    seconds_elapsed_since,
    timeout,
)
from galini.math import is_close, mc


logger = get_logger(__name__)

coramin_logger = coramin_obbt.logger  # pylint: disable=invalid-name
coramin_logger.disabled = True


def domain_adjusted_lower_bound(domain, a, b):
    """Return the maximum between a and b, rounded if the domain is integer."""
    return _domain_adjusted_bound(domain, a, b, max, np.floor, np.ceil)


def domain_adjusted_upper_bound(domain, a, b):
    """Return the minimum between a and b, rounded if the domain is integer."""
    return _domain_adjusted_bound(domain, a, b, min, np.ceil, np.floor)


def _domain_adjusted_bound(domain, a, b, pick_bound, round_bound,
                           round_bound_away):
    if b is None:
        bound = a
    elif a is not None:
        bound = pick_bound(a, b)
    else:
        return None

    if domain.is_integer() and bound is not None:
        if is_close(round_bound(bound), bound, atol=mc.epsilon, rtol=0.0):
            return round_bound(bound)
        return round_bound_away(bound)

    return bound


def perform_fbbt_on_problem(run_id, problem, fbbt_maxiter, fbbt_timelimit):
    """Perform FBBT on problem."""
    logger.debug(run_id, 'Performing FBBT')
    try:
        bounds = perform_fbbt(
            problem,
            maxiter=fbbt_maxiter,
            timelimit=fbbt_timelimit,
        )

        bounds, monotonicity, convexity = \
            propagate_special_structure(problem, bounds)
        return bounds, monotonicity, convexity

    # pylint: disable=broad-except
    except Exception as ex:
        logger.warning(run_id, 'FBBT Failed: {}', str(ex))
        bounds, monotonicity, convexity = \
            propagate_special_structure(problem)
        return bounds, monotonicity, convexity


def tighten_bounds_on_problem(run_id, problem, bounds):
    """Set bounds on problem only if they don't cause infeasibility."""
    logger.debug(run_id, 'Set FBBT Bounds')
    cause_infeasibility = None
    for v in problem.variables:
        vv = problem.variable_view(v)
        new_bound = bounds[v]
        if new_bound is None:
            new_bound = Interval(None, None)

        new_lb = domain_adjusted_lower_bound(
            v.domain,
            new_bound.lower_bound,
            vv.lower_bound()
        )

        new_ub = domain_adjusted_upper_bound(
            v.domain,
            new_bound.upper_bound,
            vv.upper_bound()
        )

        if new_lb > new_ub:
            cause_infeasibility = v

    if cause_infeasibility is not None:
        logger.info(
            run_id, 'Bounds on variable {} cause infeasibility',
            cause_infeasibility.name
        )
    else:
        for v in problem.variables:
            vv = problem.variable_view(v)
            new_bound = bounds[v]

            if new_bound is None:
                new_bound = Interval(None, None)

            new_lb = domain_adjusted_lower_bound(
                v.domain,
                new_bound.lower_bound,
                vv.lower_bound()
            )

            new_ub = domain_adjusted_upper_bound(
                v.domain,
                new_bound.upper_bound,
                vv.upper_bound()
            )

            if np.isinf(new_lb):
                new_lb = -np.inf

            if np.isinf(new_ub):
                new_ub = np.inf

            if np.abs(new_ub - new_lb) < mc.epsilon:
                new_lb = new_ub

            logger.debug(run_id, '  {}: [{}, {}]', v.name, new_lb, new_ub)
            vv.set_lower_bound(new_lb)
            vv.set_upper_bound(new_ub)

    # group_name = '_'.join([str(c) for c in node.coordinate])
    # logger.tensor(run_id, group_name, 'lb', problem.lower_bounds)
    # logger.tensor(run_id, group_name, 'ub', problem.upper_bounds)


def perform_obbt_on_model(model, problem, obbt_timelimit, obbt_simplex_maxiter):
    #  obbt_timelimit = self.solver.config['obbt_timelimit']
    obbt_start_time = current_time()

    for var in model.component_data_objects(ctype=pe.Var):
        var.domain = pe.Reals

        if not (var.lb is None or np.isfinite(var.lb)):
            var.setlb(None)

        if not (var.ub is None or np.isfinite(var.ub)):
            var.setub(None)

    relaxed_model = relax(model)

    for obj in relaxed_model.component_data_objects(ctype=pe.Objective):
        relaxed_model.del_component(obj)

    solver = pe.SolverFactory('cplex_persistent')
    solver.set_instance(relaxed_model)
    # TODO(fra): make this non-cplex specific
    simplex_limits = solver._solver_model.parameters.simplex.limits # pylint: disable=protected-access
    simplex_limits.iterations.set(obbt_simplex_maxiter)
    # collect variables in nonlinear constraints
    nonlinear_variables = ComponentSet()
    for constraint in model.component_data_objects(ctype=pe.Constraint):
        # skip linear constraint
        if constraint.body.polynomial_degree() == 1:
            continue

        for var in identify_variables(constraint.body,
                                      include_fixed=False):
            # Coramin will complain about variables that are fixed
            # Note: Coramin uses an hard-coded 1e-6 tolerance
            if var.lb is None or var.ub is None:
                nonlinear_variables.add(var)
            else:
                if not var.ub - var.lb < 1e-6:
                    nonlinear_variables.add(var)

    relaxed_vars = [
        getattr(relaxed_model, v.name)
        for v in nonlinear_variables
    ]

    logger.info(0, 'Performing OBBT on {} variables', len(relaxed_vars))

    time_left = obbt_timelimit - seconds_elapsed_since(obbt_start_time)
    with timeout(time_left, 'Timeout in OBBT'):
        result = coramin_obbt.perform_obbt(
            relaxed_model, solver, relaxed_vars
        )

    if result is None:
        return

    logger.debug(0, 'New Bounds')
    for v, new_lb, new_ub in zip(relaxed_vars, *result):
        vv = problem.variable_view(v.name)
        if new_lb is None or new_ub is None:
            logger.warning(0, 'Could not tighten variable {}', v.name)
        old_lb = vv.lower_bound()
        old_ub = vv.upper_bound()
        new_lb = domain_adjusted_lower_bound(vv.domain, new_lb, old_lb)
        new_ub = domain_adjusted_upper_bound(vv.domain, new_ub, old_ub)
        if new_lb is not None and new_ub is not None:
            if is_close(new_lb, new_ub, atol=mc.epsilon):
                if old_lb is not None and \
                        is_close(new_lb, old_lb, atol=mc.epsilon):
                    new_ub = new_lb
                else:
                    new_lb = new_ub
        vv.set_lower_bound(new_lb)
        vv.set_upper_bound(new_ub)

        logger.debug(
            0, '  {}: [{}, {}]',
            v.name, vv.lower_bound(), vv.upper_bound()
        )

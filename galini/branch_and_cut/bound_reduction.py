#  Copyright 2020 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""OBBT step of branch-and-cut algorithm."""

import coramin.domain_reduction.obbt as coramin_obbt
import numpy as np
import pyomo.environ as pe
from coramin.relaxations.auto_relax import relax
from pyomo.core.expr.current import identify_variables
from pyomo.core.kernel.component_set import ComponentSet

from galini.logging import get_logger
from galini.math import mc, is_close
from galini.timelimit import (
    current_time,
    seconds_elapsed_since,
    timeout,
)

logger = get_logger(__name__)

coramin_logger = coramin_obbt.logger  # pylint: disable=invalid-name
coramin_logger.disabled = True


def perform_obbt_on_model(model, problem, upper_bound, timelimit,
                          simplex_maxiter):
    """Perform OBBT on Pyomo model using Coramin.

    Parameters
    ----------
    model : ConcreteModel
        the pyomo concrete model
    problem : Problem
        the GALINI problem
    upper_bound : float or None
        the objective value upper bound, if known
    timelimit : int
        a timelimit, in seconds
    simplex_maxiter : int
        the maximum number of simplex iterations

    """
    obbt_start_time = current_time()

    for var in model.component_data_objects(ctype=pe.Var):
        var.domain = pe.Reals

        if not (var.lb is None or np.isfinite(var.lb)):
            var.setlb(None)

        if not (var.ub is None or np.isfinite(var.ub)):
            var.setub(None)

    relaxed_model = relax(model)

    solver = pe.SolverFactory('cplex_persistent')
    solver.set_instance(relaxed_model)
    # TODO(fra): make this non-cplex specific
    simplex_limits = solver._solver_model.parameters.simplex.limits  # pylint: disable=protected-access
    simplex_limits.iterations.set(simplex_maxiter)
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
            if not var.has_lb() or not var.has_ub():
                nonlinear_variables.add(var)
            else:
                if not np.abs(var.ub - var.lb) < 1e-6:
                    nonlinear_variables.add(var)

    relaxed_vars = [
        getattr(relaxed_model, v.name)
        for v in nonlinear_variables
    ]

    logger.info(0, 'Performing OBBT on {} variables', len(relaxed_vars))

    # Avoid Coramin raising an exception if the problem has no objective
    # value but we set an upper bound.
    objectives = model.component_data_objects(
        pe.Objective, active=True, sort=True, descend_into=True
    )
    if len(list(objectives)) == 0:
        upper_bound = None

    time_left = timelimit - seconds_elapsed_since(obbt_start_time)
    with timeout(time_left, 'Timeout in OBBT'):
        result = coramin_obbt.perform_obbt(
            relaxed_model, solver,
            varlist=relaxed_vars,
            objective_bound=upper_bound
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
        new_lb = best_lower_bound(vv.domain, new_lb, old_lb)
        new_ub = best_upper_bound(vv.domain, new_ub, old_ub)
        if not new_lb is None and not new_ub is None:
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


def best_lower_bound(domain, a, b):
    """Returns the best lower bound between `a` and `b`.

    Parameters
    ----------
    domain : Domain
        the variable domain
    a : float or None
    b : float or None
    """
    if b is None:
        lb = a
    elif a is not None:
        lb = max(a, b)
    else:
        return None

    if domain.is_integer() and lb is not None:
        if is_close(np.floor(lb), lb, atol=mc.epsilon, rtol=0.0):
            return np.floor(lb)
        return np.ceil(lb)

    return lb


def best_upper_bound(domain, a, b):
    """Returns the best upper bound between `a` and `b`.

    Parameters
    ----------
    domain : Domain
        the variable domain
    a : float or None
    b : float or None
    """
    if b is None:
        ub = a
    elif a is not None:
        ub = min(a, b)
    else:
        return None

    if domain.is_integer() and ub is not None:
        if is_close(np.ceil(ub), ub, atol=mc.epsilon, rtol=0.0):
            return np.ceil(ub)
        return np.floor(ub)

    return ub

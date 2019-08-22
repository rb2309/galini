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

"""Branch & Cut algorithm termination."""

from enum import Enum
from galini.timelimit import seconds_left
from galini.quantities import relative_gap, absolute_gap
from galini.math import is_close


class TerminationReason(Enum):
    CONVERGED = 0
    TIMEOUT = 1
    ITERATIONS_EXCEEDED = 2

    def is_converged(self):
        return self == self.CONVERGED

    def is_timeout(self):
        return self == self.TIMEOUT

    def is_iterations_exceeded(self):
        return self == self.ITERATIONS_EXCEEDED


def has_converged(state, condition, paranoid_mode=False):
    """Return true if converged."""
    rel_gap = relative_gap(state.lower_bound, state.upper_bound)
    abs_gap = absolute_gap(state.lower_bound, state.upper_bound)

    bounds_close = is_close(
        state.lower_bound,
        state.upper_bound,
        rtol=condition.relative_tolerance,
        atol=condition.tolerance,
    )

    if paranoid_mode:
        assert (state.lower_bound <= state.upper_bound or bounds_close)

    return (
        rel_gap <= condition.relative_tolerance or
        abs_gap <= condition.tolerance
    )


def has_timeout():
    """Return true if it exceeded the time limit."""
    return seconds_left() < 0


def has_exceeded_iterations(state, condition):
    """Return true if the number of iterations/visited nodes was exceeded."""
    return state.nodes_visited > condition.node_limit


def should_terminate(state, condition, paranoid_mode=False):
    """Return termination reason if should terminate, None otherwise."""
    if has_converged(state, condition, paranoid_mode):
        return TerminationReason.CONVERGED

    if has_timeout():
        return TerminationReason.TIMEOUT

    if has_exceeded_iterations(state, condition):
        return TerminationReason.ITERATIONS_EXCEEDED

    return None

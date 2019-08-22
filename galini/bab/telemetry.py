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


"""Collect and output branch & cut telemetry information."""

import numpy as np


class BranchAndCutTelemetry:
    def __init__(self, telemetry):
        self._telemetry = telemetry

        self._upper_bound = \
            self._telemetry.create_gauge('bac.upper_bound', np.inf)
        self._lower_bound = \
            self._telemetry.create_gauge('bac.lower_bound', -np.inf)
        self._visited_nodes_counter = \
            self._telemetry.create_counter('bac.visited_nodes', 0)

        self._iteration = 0

    def end_iteration(self, run_id, tree):
        self._upper_bound.set_value(tree.upper_bound)
        self._lower_bound.set_value(tree.lower_bound)
        self._visited_nodes_counter.increment(1)
        self._telemetry.log_at_end_of_iteration(run_id, self._iteration)
        self._iteration += 1

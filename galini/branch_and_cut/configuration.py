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


_configuration_keys = [
    'obbt_timelimit',
    'obbt_simplex_maxiter',

    'fbbt_maxiter',
    'fbbt_timelimit',
    'fbbt_max_quadratic_size',
    'fbbt_max_expr_children',

    'root_node_feasible_solution_seed',
    'root_node_feasible_solution_search_timelimit',

    'catch_keyboard_interrupt',
]


class BranchAndCutConfiguration:
    def __init__(self, config, paranoid_mode):
        self.paranoid_mode = paranoid_mode
        self.termination = TerminationConfiguration(config)
        self.cuts = CutsConfiguration(config)

        for key in _configuration_keys:
            value = getattr(config, key)
            setattr(self, key, value)


class TerminationConfiguration:
    def __init__(self, config):
        self.tolerance = config['tolerance']
        self.relative_tolerance = config['relative_tolerance']
        self.node_limit = config['node_limit']


class CutsConfiguration:
    def __init__(self, config):
        config = config['branch_and_cut']
        self.maxiter = config['maxiter']
        self.use_milp_relaxation = config['use_milp_relaxation']
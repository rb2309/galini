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

"""Branch problem at a point."""

from galini.core import VariableView


def branch_at_point(problem, branching_point):
    """Branch problem at branching_point, returning the new child problems."""
    branching_var = branching_point.variable
    if isinstance(branching_var, VariableView):
        branching_var = branching_var.variable

    var = problem.variable_view(branching_var.idx)
    for point in branching_point.points:
        if point < var.lower_bound() or point > var.upper_bound():
            raise RuntimeError('Branching outside variable bounds')

    children = []

    new_upper_bound = var.lower_bound()
    for point in branching_point.points:
        new_lower_bound = new_upper_bound
        new_upper_bound = point

        child = _create_child_problem(
            problem, var, new_lower_bound, new_upper_bound
        )
        children.append(child)

    child = _create_child_problem(
        problem, var, new_upper_bound, var.upper_bound()
    )
    children.append(child)

    return children


def _create_child_problem(problem, var, new_lower_bound, new_upper_bound):
    child = problem.make_child()
    child_var = child.variable_view(var.idx)
    child_var.set_lower_bound(new_lower_bound)
    child_var.set_upper_bound(new_upper_bound)
    return child
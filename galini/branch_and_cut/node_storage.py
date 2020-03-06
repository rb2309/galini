#  Copyright 2019 Francesco Ceccon
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

"""Branch & Cut node storage. Contains original and convex problem."""
import pyomo.environ as pe
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from coramin.relaxations import relaxation_data_objects
from galini.pyomo import safe_setlb, safe_setub
from galini.branch_and_bound.branching import branch_at_point
from galini.cuts.pool import CutNodeStorage, CutPool
from galini.relaxations.relax import relax


class BranchingDecision:
    __slots__ = ('variable', 'point')

    def __init__(self, variable, point):
        self.variable = variable
        self.point = point

    def __str__(self):
        return 'BranchingDecision(variable={}, point={})'.format(
            self.variable, self.point
        )



class _NodeStorageBase:
    def __init__(self, root, parent, bounds):
        self.root = root
        self.parent = parent
        self._bounds = bounds
        self.branching_decision = None
        self.branching_point = None
        self.cut_pool = None
        self.cut_node_storage = None

    @property
    def is_root(self):
        return None

    @property
    def model_bounds(self):
        return self._bounds

    def branching_data(self):
        return self.model()

    def model(self):
        for var, (lb, ub) in self._bounds.items():
            safe_setlb(var, lb)
            safe_setub(var, ub)
        return self.root._model

    def convex_model(self):
        # TODO(fra): really use a convex model
        return None

    def linear_model(self):
        linear_model = self.root._linear_model
        for var, (lb, ub) in self._bounds.items():
            # TODO(fra): proper map between vars
            linear_var = getattr(linear_model, var.name)
            safe_setlb(linear_var, lb)
            safe_setub(linear_var, ub)

        for relaxation in relaxation_data_objects(linear_model, active=True, descend_into=True):
            aux_var = relaxation.get_aux_var()
            rhs_expr = relaxation.get_rhs_expr()
            new_lb, new_ub = compute_bounds_on_expr(rhs_expr)
            safe_setlb(aux_var, new_lb)
            safe_setub(aux_var, new_ub)
            relaxation.add_cut()
            relaxation.rebuild()
        return linear_model

    def branch_at_point(self, branching_point):
        assert self.branching_point is None
        self.branching_point = branching_point

        children_bounds = branch_at_point(self.root._model, self._bounds, branching_point)

        return [
            NodeStorage(self.root, self, bounds, branching_point.variable)
            for bounds in children_bounds
        ]


class NodeStorage(_NodeStorageBase):
    def __init__(self, root, parent, bounds, branching_variable):
        super().__init__(root, parent, bounds)
        self.cut_pool = parent.cut_pool
        self.cut_node_storage = \
            CutNodeStorage(parent.cut_node_storage, parent.cut_pool)
        self.branching_variable = branching_variable

    @property
    def is_root(self):
        return False


class RootNodeStorage(_NodeStorageBase):
    def __init__(self, model):
        bounds = pe.ComponentMap(
            (var, var.bounds) for var in model.component_data_objects(pe.Var, active=True)
        )
        super().__init__(root=self, parent=None, bounds=bounds)
        self._model = model
        self._convex_model = None
        self._linear_model = None
        self._aux_var_relaxation_map = None
        self.cut_pool = CutPool(model)
        self.cut_node_storage = CutNodeStorage(None, self.cut_pool)

    def model(self):
        return self._model

    def convex_model(self):
        # TODO(fra): really use a convex model
        return None

    def linear_model(self):
        if self._linear_model is not None:
            return self._linear_model
        linear_model, aux_var_relaxation_map = relax(self._model)
        self._linear_model = linear_model
        self._aux_var_relaxation_map = aux_var_relaxation_map
        return self._linear_model


    @property
    def is_root(self):
        return True

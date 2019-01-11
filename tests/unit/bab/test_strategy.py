# pylint: skip-file
import pytest
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.bab.tree import BabTree
from galini.bab.strategy import KSectionBranchingStrategy


class FakeSelectionStrategy:
    def insert_node(self, node):
        pass


@pytest.fixture()
def problem():
    m = aml.ConcreteModel()
    m.I = range(5)
    m.x = aml.Var(m.I, bounds=(-1, 2))
    m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))
    return dag_from_pyomo_model(m)


class TestKFoldBranchingStrategy:
    def test_bisect(self, problem):
        bisect_strat = KSectionBranchingStrategy()
        tree = BabTree(bisect_strat, FakeSelectionStrategy())
        tree.add_root(problem)
        node = tree.root
        for i in range(5):
            children = node.branch(bisect_strat)
            assert len(children) == 2
            for child in children:
                assert child.variable.idx == i
            node = children[0]

    def test_ksection(self, problem):
        ksection_strat = KSectionBranchingStrategy(7)
        tree = BabTree(ksection_strat, FakeSelectionStrategy())
        tree.add_root(problem)
        node = tree.root
        for i in range(5):
            children = node.branch(ksection_strat)
            assert len(children) == 7
            for child in children:
                assert child.variable.idx == i
            node = children[0]

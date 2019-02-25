# pylint: skip-file
import pytest
import pyomo.environ as aml
from galini.core import ExpressionTransformation
from galini.pyomo import dag_from_pyomo_model


def create_problem():
    m = aml.ConcreteModel()
    m.I = range(10)
    m.x = aml.Var(m.I, bounds=(-1, 2))
    m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))
    m.cons = aml.Constraint(m.I[1:], rule=lambda m, i: aml.cos(m.x[0]) * aml.sin(m.x[i]) >= 0)
    return dag_from_pyomo_model(m)


@pytest.fixture()
def problem():
    return create_problem()


class MyTransformation(ExpressionTransformation):
    def __init__(self):
        ExpressionTransformation.__init__(self)
        self.count = 0

    def transform(self, expr):
        self.count += 1
        print(self.count, expr)
        return expr


class TestRelaxedProblem:
    def test_start_with_original_variables(self, problem):
        relaxation = problem.make_relaxed("test", MyTransformation())
        assert 10 == relaxation.num_variables
        assert 0 == relaxation.num_objectives
        assert 0 == relaxation.num_constraints

    def test_call_transform(self, problem):
        transform = MyTransformation()
        relaxation = problem.make_relaxed("test", transform)
        assert 10 == relaxation.num_variables
        assert 0 == relaxation.num_objectives
        assert 0 == relaxation.num_constraints

        for cons in problem.constraints:
            relaxation.add_constraint(
                cons.name,
                cons.root_expr,
                cons.lower_bound,
                cons.upper_bound,
            )

        assert 9 == relaxation.num_constraints
        assert 9 == transform.count

        for obj in problem.objectives:
            relaxation.add_objective(
                obj.name,
                obj.root_expr,
                obj.sense
            )

        assert 1 == relaxation.num_objectives
        assert 10 == transform.count

        assert len(problem.vertices) == len(relaxation.vertices)

        for vertex in relaxation.vertices:
            assert relaxation == vertex.problem

        for vertex in problem.vertices:
            assert problem == vertex.problem

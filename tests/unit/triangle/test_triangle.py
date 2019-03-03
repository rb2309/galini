import pytest
import numpy as np
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.solvers import SolversRegistry
from galini.config import ConfigurationManager
from galini.cuts import CutsGeneratorsRegistry
from galini.abb.relaxation import AlphaBBRelaxation
from galini.bab.branch_and_cut import BranchAndCutAlgorithm
from galini.core import Constraint
from galini.triangle_cuts.generator import TriangleCutsGenerator


@pytest.fixture()
def problem():
    Q = [[28.0, 23.0, 0.0, 0.0, 0.0, 2.0, 0.0, 24.0],
         [23.0, 0.0, -23.0, -44.0, 10.0, 0.0, 7.0, -7.0],
         [0.0, -23.0, 18.0, 41.0, 0.0, -3.0, -5.0, 2.0],
         [0.0, -44.0, 41.0, -5.0, 5.0, -1.0, 16.0, -50.0],
         [0.0, 10.0, 0.0, 5.0, 0.0, -2.0, -4.0, 21.0],
         [2.0, 0.0, -3.0, -1.0, -2.0, 34.0, -9.0, 20.0],
         [0.0, 7.0, -5.0, 16.0, -4.0, -9.0, 0.0, 0.0],
         [24.0, -7.0, 2.0, -50.0, 21.0, 20.0, 0.0, -45.0]]

    C = [-44, -48, 10, 45, 0, 2, 3, 4, 5]

    Qc = [
        [-28, 13, 5],
        [13, 0, 0],
        [0, 0, 0],
    ]

    m = aml.ConcreteModel("model_1")
    m.I = range(8)
    m.x = aml.Var(m.I, bounds=(0, 1))
    m.f = aml.Objective(
        expr=sum(-Q[i][j] * m.x[i] * m.x[j] for i in m.I for j in m.I) + sum(-C[i] * m.x[i] for i in m.I))
    m.c = aml.Constraint(expr=sum(Qc[i][j] * m.x[i] * m.x[j] for i in m.I[0:3] for j in m.I[0:3]) >= -10)

    return dag_from_pyomo_model(m)


def test_triangle_cuts(problem):
    solvers_reg = SolversRegistry()
    solver_cls = solvers_reg.get('ipopt')
    cuts_gen_reg = CutsGeneratorsRegistry()
    config_manager = ConfigurationManager()
    config_manager.initialize(solvers_reg, cuts_gen_reg)
    config = config_manager.configuration
    config.update({
        'cuts_generator': {
            'triangle': {
                'selection_size': 2,
                'min_tri_cuts_per_round': 0,
            },
        }
    })
    solver_ipopt = solver_cls(config, solvers_reg, cuts_gen_reg)
    solver_mip = solver_ipopt.instantiate_solver("mip")

    # Test adjacency matrix
    triangle_cuts_gen = TriangleCutsGenerator(config.cuts_generator.triangle)
    triangle_cuts_gen.before_start_at_root(problem)
    assert (np.allclose(triangle_cuts_gen._get_adjacency_matrix(problem),
                        [
                            [1, 1, 1, 0, 0, 1, 0, 1],
                            [1, 0, 1, 1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 0, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 0, 1, 0, 1, 1, 1],
                            [1, 0, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0, 1]
                        ]))

    # Test triangle cut violations
    relaxation = AlphaBBRelaxation()
    relaxed_problem = relaxation.relax(problem)
    mip_solution = solver_mip.solve(relaxed_problem, logger=None)
    assert mip_solution.status.is_success()
    assert (np.allclose(triangle_cuts_gen._get_triangle_violations(relaxed_problem, mip_solution),
                        [[0, 0, 0.5], [0, 1, -0.5], [0, 2, -0.5], [0, 3, -0.5], [1, 0, 0.5], [1, 1, -0.5], [1, 2, -0.5],
                         [1, 3, -0.5], [2, 0, 0.0], [2, 1, 0.0], [2, 2, -0.5], [2, 3, -0.5], [3, 0, 0.0], [3, 1, 0.0],
                         [3, 2, 0.0], [3, 3, -1.0], [4, 0, 0.0], [4, 1, -0.5], [4, 2, 0.0], [4, 3, -0.5], [5, 0, -1.0],
                         [5, 1, 0.0], [5, 2, 0.0], [5, 3, 0.0], [6, 0, 0.0], [6, 1, -1.0], [6, 2, 0.0], [6, 3, 0.0],
                         [7, 0, -1.0], [7, 1, 0.0], [7, 2, 0.0], [7, 3, 0.0], [8, 0, -0.5], [8, 1, -0.5], [8, 2, 0.5],
                         [8, 3, -0.5], [9, 0, -0.5], [9, 1, -0.5], [9, 2, 0.5], [9, 3, -0.5], [10, 0, -0.5],
                         [10, 1, -0.5], [10, 2, -0.5], [10, 3, 0.5], [11, 0, 0.5], [11, 1, -0.5], [11, 2, -0.5],
                         [11, 3, -0.5], [12, 0, -0.5], [12, 1, 0.5], [12, 2, -0.5], [12, 3, -0.5], [13, 0, 0.0],
                         [13, 1, 0.0], [13, 2, -0.5], [13, 3, -0.5], [14, 0, -0.5], [14, 1, 0.5], [14, 2, -0.5],
                         [14, 3, -0.5], [15, 0, 0.5], [15, 1, -0.5], [15, 2, -0.5], [15, 3, -0.5], [16, 0, -0.5],
                         [16, 1, 0.0], [16, 2, -0.5], [16, 3, 0.0], [17, 0, 0.0], [17, 1, -0.5], [17, 2, 0.0],
                         [17, 3, -0.5], [18, 0, 0.0], [18, 1, 0.0], [18, 2, -0.5], [18, 3, -0.5], [19, 0, 0.5],
                         [19, 1, -0.5], [19, 2, -0.5], [19, 3, -0.5], [20, 0, -0.5], [20, 1, 0.5], [20, 2, -0.5],
                         [20, 3, -0.5], [21, 0, 0.0], [21, 1, -0.5], [21, 2, 0.0], [21, 3, -0.5], [22, 0, -0.5],
                         [22, 1, 0.0], [22, 2, -0.5], [22, 3, 0.0], [23, 0, -0.5], [23, 1, 0.0], [23, 2, -0.5],
                         [23, 3, 0.0], [24, 0, 0.0], [24, 1, -0.5], [24, 2, 0.0], [24, 3, -0.5]]))

    # Test at root node
    algo = BranchAndCutAlgorithm(solver_ipopt, solver_mip, triangle_cuts_gen, config)
    algo._cuts_generators_manager.before_start_at_root(problem)
    algo._cuts_generators_manager.before_start_at_node(problem)
    relaxed_problem = relaxation.relax(problem)
    nbs_cuts = []
    mip_sols = []
    for iteration in range(5):
        mip_solution = solver_mip.solve(relaxed_problem, logger=None)
        assert mip_solution.status.is_success()
        mip_sols.append(mip_solution.objectives[0].value)
        # Generate new cuts
        new_cuts = algo._cuts_generators_manager.generate(problem, relaxed_problem, mip_solution, None, None)
        # Add cuts as constraints
        nbs_cuts.append(len(list(new_cuts)))
        for cut in new_cuts:
            new_cons = Constraint(cut.name, cut.expr, cut.lower_bound, cut.upper_bound)
            relaxation._relax_constraint(problem, relaxed_problem, new_cons)
    assert (nbs_cuts == [2, 2, 2, 0, 0])
    assert (np.allclose(mip_sols, [-200.0, -196.85714285714283, -196.5, -196.0, -196.0]))

    # Test when branched on x0 in [0.5, 1]
    x0 = problem.variable_view(problem.variables[0])
    x0.set_lower_bound(0.5)
    relaxed_problem = relaxation.relax(problem)
    algo._cuts_generators_manager.before_start_at_node(problem)
    mip_sols = []
    mip_solution = None
    for iteration in range(5):
        mip_solution = solver_mip.solve(relaxed_problem, logger=None)
        assert mip_solution.status.is_success()
        mip_sols.append(mip_solution.objectives[0].value)
        # Generate new cuts
        new_cuts = algo._cuts_generators_manager.generate(problem, relaxed_problem, mip_solution, None, None)
        # Add cuts as constraints
        for cut in new_cuts:
            new_cons = Constraint(cut.name, cut.expr, cut.lower_bound, cut.upper_bound)
            relaxation._relax_constraint(problem, relaxed_problem, new_cons)
    assert(np.allclose(mip_sols,
           [-193.88095238095238, -187.96808510638297, -187.42857142857147, -187.10869565217394, -187.10869565217394]))
    triangle_cuts_gen.after_end_at_node(problem, mip_solution)
    triangle_cuts_gen.after_end_at_root(problem, mip_solution)

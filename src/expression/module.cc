/* Copyright 2018 Francesco Ceccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================== */
#include "module.h"

#include <pybind11/stl.h>

#include "ad/expression_tree_data.h"
#include "problem/problem_base.h"

namespace py = pybind11;

namespace galini {

namespace expression {

void init_module(py::module& m) {
  auto suspect = py::module::import("suspect");
  auto suspect_expression = suspect.attr("expression");
  auto ExpressionType = suspect_expression.attr("ExpressionType");
  auto UnaryFunctionType = suspect_expression.attr("UnaryFunctionType");

  py::class_<Expression, Expression::ptr>(m, "Expression")
    .def_property_readonly("problem", &Expression::problem)
    .def_property_readonly("graph", &Expression::graph)
    .def_property_readonly("idx", &Expression::idx)
    .def_property_readonly("uid", &Expression::uid)
    .def_property("depth", &Expression::depth, &Expression::set_depth)
    .def_property_readonly("default_depth", &Expression::default_depth)
    .def_property_readonly("num_children", &Expression::num_children)
    .def_property_readonly("children", &Expression::children)
    .def_property_readonly("args", &Expression::children)
    .def("expression_tree_data", &Expression::expression_tree_data, py::arg("num_variables") = 0)
    .def("nargs", [](const Expression &ex) { return ex.num_children(); })
    .def("polynomial_degree", &Expression::polynomial_degree)
    .def("nth_children", &Expression::nth_children)
    .def("is_constant", &Expression::is_constant)
    .def("is_expression_type", &Expression::is_expression)
    .def("is_variable", &Expression::is_variable)
    .def("is_variable_type", &Expression::is_variable);

  py::class_<UnaryExpression, Expression, UnaryExpression::ptr>(m, "UnaryExpression");
  py::class_<BinaryExpression, Expression, BinaryExpression::ptr>(m, "BinaryExpression");
  py::class_<NaryExpression, Expression, NaryExpression::ptr>(m, "NaryExpression");

  py::class_<Variable, Expression, Variable::ptr>(m, "Variable")
    .def(py::init<const std::string&, py::object, py::object, py::object>())
    .def(py::init<const Expression::problem_ptr&, const std::string&, py::object, py::object, py::object>())
    .def_property("reference", &Variable::reference, &Variable::set_reference)
    .def_property_readonly("name", &Variable::name)
    .def_property_readonly("lower_bound", &Variable::lower_bound)
    .def_property_readonly("upper_bound", &Variable::upper_bound)
    .def_property_readonly("lb", &Variable::lower_bound)
    .def_property_readonly("ub", &Variable::upper_bound)
    .def_property_readonly("domain", &Variable::domain)
    .def_property_readonly("is_auxiliary", &Variable::is_auxiliary)
    .def_property_readonly("expression_type",
			   [ExpressionType](const Variable&) { return ExpressionType.attr("Variable"); });

  py::class_<Constant, Expression, Constant::ptr>(m, "Constant")
    .def(py::init<double>())
    .def(py::init<const Expression::problem_ptr&, double>())
    .def_property_readonly("value", &Constant::value)
    .def_property_readonly("expression_type",
			   [ExpressionType](const Constant&) { return ExpressionType.attr("Constant"); });


  py::class_<NegationExpression, UnaryExpression, NegationExpression::ptr>(m, "NegationExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const NegationExpression&) { return ExpressionType.attr("Negation"); });

  py::class_<UnaryFunctionExpression, UnaryExpression,
	     UnaryFunctionExpression::ptr>(m, "UnaryFunctionExpression")
    .def_property_readonly("expression_type",
			   [ExpressionType](const UnaryFunctionExpression&) { return ExpressionType.attr("UnaryFunction"); });

  py::class_<AbsExpression, UnaryFunctionExpression, AbsExpression::ptr>(m, "AbsExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const AbsExpression&) { return UnaryFunctionType.attr("Abs"); });


  py::class_<SqrtExpression, UnaryFunctionExpression, SqrtExpression::ptr>(m, "SqrtExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const SqrtExpression&) { return UnaryFunctionType.attr("Sqrt"); });

  py::class_<ExpExpression, UnaryFunctionExpression, ExpExpression::ptr>(m, "ExpExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const ExpExpression&) { return UnaryFunctionType.attr("Exp"); });

  py::class_<LogExpression, UnaryFunctionExpression, LogExpression::ptr>(m, "LogExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const LogExpression&) { return UnaryFunctionType.attr("Log"); });

  py::class_<SinExpression, UnaryFunctionExpression, SinExpression::ptr>(m, "SinExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const SinExpression&) { return UnaryFunctionType.attr("Sin"); });

  py::class_<CosExpression, UnaryFunctionExpression, CosExpression::ptr>(m, "CosExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const CosExpression&) { return UnaryFunctionType.attr("Cos"); });

  py::class_<TanExpression, UnaryFunctionExpression, TanExpression::ptr>(m, "TanExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const TanExpression&) { return UnaryFunctionType.attr("Tan"); });

  py::class_<AsinExpression, UnaryFunctionExpression, AsinExpression::ptr>(m, "AsinExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const AsinExpression&) { return UnaryFunctionType.attr("Asin"); });

  py::class_<AcosExpression, UnaryFunctionExpression, AcosExpression::ptr>(m, "AcosExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const AcosExpression&) { return UnaryFunctionType.attr("Acos"); });

  py::class_<AtanExpression, UnaryFunctionExpression, AtanExpression::ptr>(m, "AtanExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const AtanExpression&) { return UnaryFunctionType.attr("Atan"); });

  py::class_<ProductExpression, BinaryExpression, ProductExpression::ptr>(m, "ProductExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const ProductExpression&) { return ExpressionType.attr("Product"); });

  py::class_<DivisionExpression, BinaryExpression, DivisionExpression::ptr>(m, "DivisionExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const DivisionExpression&) { return ExpressionType.attr("Division"); });

  py::class_<PowExpression, BinaryExpression, PowExpression::ptr>(m, "PowExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const PowExpression&) { return ExpressionType.attr("Power"); });

  py::class_<SumExpression, NaryExpression, SumExpression::ptr>(m, "SumExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const SumExpression&) { return ExpressionType.attr("Sum"); });

  py::class_<LinearExpression, NaryExpression, LinearExpression::ptr>(m, "LinearExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&,
	 const std::vector<double>&, double>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&,
	 const std::vector<double>&, double>())
    .def(py::init<const std::vector<LinearExpression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<LinearExpression::ptr>&>())
    .def("coefficient", &LinearExpression::coefficient)
    .def_property_readonly("constant_term", &LinearExpression::constant)
    .def_property_readonly("constant", &LinearExpression::constant)
    .def_property_readonly("linear_vars", &Expression::children)
    .def_property_readonly("linear_coefs", &LinearExpression::linear_coefs)
    .def_property_readonly("expression_type",
			   [ExpressionType](const LinearExpression&) { return ExpressionType.attr("Linear"); });

  py::class_<QuadraticExpression, NaryExpression, QuadraticExpression::ptr>(m, "QuadraticExpression")
    .def(py::init<const std::vector<typename Expression::ptr>&,
	 const std::vector<typename Expression::ptr>&, const std::vector<double>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<typename Expression::ptr>&,
	 const std::vector<typename Expression::ptr>&, const std::vector<double>&>())
    .def(py::init<const std::vector<QuadraticExpression::ptr>&>())
    .def(py::init<const Expression::problem_ptr&, const std::vector<QuadraticExpression::ptr>&>())
    .def("coefficient", &QuadraticExpression::coefficient)
    .def_property_readonly("terms", &QuadraticExpression::terms)
    .def_property_readonly("expression_type",
			   [ExpressionType](const QuadraticExpression&) { return ExpressionType.attr("Quadratic"); });

  py::class_<BilinearTerm>(m, "BilinearTerm")
    .def_readonly("var1", &BilinearTerm::var1)
    .def_readonly("var2", &BilinearTerm::var2)
    .def_readonly("coefficient", &BilinearTerm::coefficient);

  py::class_<Graph, Graph::ptr>(m, "Graph")
    .def(py::init())
    .def("insert_tree", &Graph::insert_tree)
    .def("insert_vertex", &Graph::insert_vertex)
    .def("expression_tree_data", &Graph::expression_tree_data)
    .def("max_depth", &Graph::max_depth)
    .def("__len__", &Graph::size)
    .def("__iter__",
	 [](const Graph &g) { return py::make_iterator(g.begin(), g.end()); },
	 py::keep_alive<0, 1>() /* keep alive while we iterate */)
    .def("__reversed__",
	 [](const Graph &g) { return py::make_iterator(g.rbegin(), g.rend()); },
	 py::keep_alive<0, 1>() /* keep alive while we iterate */);
}

} // namespace expression

} // namespace galini

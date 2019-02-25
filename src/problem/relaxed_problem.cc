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
#include "relaxed_problem.h"

#include <unordered_map>

#include "expression/expression_base.h"
#include "detail.h"


namespace galini {

namespace problem {


ExpressionTransformation::ExpressionTransformation()
  : existing_to_new_expr_() {}


std::shared_ptr<Expression> ExpressionTransformation::do_transform(const std::shared_ptr<Expression>& expr) const {
  auto new_expr = transform(expr);
  return detail::duplicate_tree(expr, existing_to_new_expr_);
}

RelaxedProblem::RelaxedProblem(const Problem::ptr& parent, const std::string &name, const std::shared_ptr<ExpressionTransformation>& transformation)
  : RootProblem(name), parent_(parent), transformation_(transformation) {}

std::shared_ptr<Constraint> RelaxedProblem::add_constraint(const std::string& name,
							   const std::shared_ptr<Expression>& expr,
							   py::object lower_bound,
							   py::object upper_bound) {
  check_vertices_belong_to_parent(expr);
  auto transformed_expr = transformation_->do_transform(expr);
  insert_tree(transformed_expr);
  return do_add_constraint(name, transformed_expr, lower_bound, upper_bound);
}

std::shared_ptr<Objective> RelaxedProblem::add_objective(const std::string& name,
							 const std::shared_ptr<Expression>& expr,
							 py::object sense) {
  check_vertices_belong_to_parent(expr);
  auto transformed_expr = transformation_->do_transform(expr);
  insert_tree(transformed_expr);
  return do_add_objective(name, transformed_expr, sense);
}

void RelaxedProblem::check_vertices_belong_to_parent(const std::shared_ptr<Expression>& expr) {
  auto vertices = detail::collect_vertices(expr);
  for (const auto& vertex : vertices) {
    auto problem = vertex->problem();
    if ((problem != nullptr) && (problem != parent_)) {
      throw std::runtime_error("RelaxedProblem only accepts new expressions or expressions that belong to its parent.");
    }
  }
}


} // namespace problem

} // namespace galini

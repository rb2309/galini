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

#include "expression/expression_base.h"


namespace galini {

namespace problem {


RelaxedProblem::RelaxedProblem(const Problem::ptr& parent, const std::string &name)
  : RootProblem(name), parent_(parent), memo_() {}


std::shared_ptr<Constraint> RelaxedProblem::add_constraint(const std::string& name,
							   const std::shared_ptr<Expression>& expr,
							   py::object lower_bound,
							   py::object upper_bound) {
  auto new_expr = duplicate_tree(expr);
  insert_tree(new_expr);
  return do_add_constraint(name, new_expr, lower_bound, upper_bound);
}

std::shared_ptr<Objective> RelaxedProblem::add_objective(const std::string& name,
							 const std::shared_ptr<Expression>& expr,
							 py::object sense) {
  auto new_expr = duplicate_tree(expr);
  insert_tree(new_expr);
  return do_add_objective(name, new_expr, sense);
}

void RelaxedProblem::duplicate_variables_from_problem(const std::shared_ptr<Problem>& other) {
  // Copy all variables to problem to keep variables indexes the same
  for (index_t i = 0; i < other->num_variables(); ++i) {
    auto var = other->variable(i);
    auto new_var = this->add_variable(var->name(),
				      other->lower_bound(var),
				      other->upper_bound(var),
				      other->domain(var));

    add_to_memo(var, new_var);

    if (var->idx() != new_var->idx()) {
      throw std::runtime_error("Index of new variable is different than original variable. This is a BUG.");
    }
  }
}

std::shared_ptr<Expression> RelaxedProblem::duplicate_tree(const std::shared_ptr<Expression>& root_expr) {
  auto vertices = this->collect_vertices(root_expr);
  for (auto it = vertices.rbegin(); it != vertices.rend(); ++it) {
    auto expr = *it;
    if ((expr->problem() != nullptr) && (expr->problem() != parent_)) {
      throw std::runtime_error("Expression belongs to wrong problem.");
    }

    if (!has_memo(expr)) {
      std::vector<Expression::ptr> children(expr->num_children());
      for (index_t i = 0; i < expr->num_children(); ++i) {
	auto child = expr->nth_children(i);
	if (!has_memo(child)) {
	  throw std::runtime_error("Error in duplicate_tree. Child is not in memo.");
	}
	children[i] = get_memo(child);
      }

      auto new_expr = expr->duplicate(children);
      add_to_memo(expr, new_expr);
    }
  }
  return get_memo(root_expr);
}

void RelaxedProblem::add_to_memo(const std::shared_ptr<Expression>& expr,
				 const std::shared_ptr<Expression>& new_expr) {
  auto expr_uid = expr->uid();
  if (memo_.find(expr_uid) != memo_.end()) {
    auto existing_expr = memo_[expr_uid];
    if (expr_uid != existing_expr->uid()) {
      return;
    }

    throw std::runtime_error("Variable already in the problem.");
  }

  memo_[expr_uid] = new_expr;
}

bool RelaxedProblem::has_memo(const std::shared_ptr<Expression>& expr) {
  return memo_.find(expr->uid()) != memo_.end();
}

std::shared_ptr<Expression> RelaxedProblem::get_memo(const std::shared_ptr<Expression>& expr) {
  return memo_.at(expr->uid());
}

} // namespace problem

} // namespace galini

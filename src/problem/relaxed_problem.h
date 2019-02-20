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
#pragma once

#include <unordered_map>

#include "expression/expression_base.h"
#include "problem/problem_base.h"
#include "problem/root_problem.h"

namespace galini {

namespace problem {

class RelaxedProblem : public RootProblem {
public:
  using ptr = std::shared_ptr<RelaxedProblem>;

  RelaxedProblem(const Problem::ptr& parent, const std::string &name);

  std::shared_ptr<Constraint> add_constraint(const std::string& name,
					     const std::shared_ptr<Expression>& expr,
					     py::object lower_bound,
					     py::object upper_bound) override;

  std::shared_ptr<Objective> add_objective(const std::string& name,
					   const std::shared_ptr<Expression>& expr,
					   py::object sense) override;

  Problem::ptr parent() {
    return parent_;
  }

  void duplicate_variables_from_problem(const std::shared_ptr<Problem>& other) override;
  //  std::vector<std::shared_ptr<Expression>> collect_vertices(const std::shared_ptr<Expression>& root_expr) override;
private:
  std::shared_ptr<Expression> duplicate_tree(const std::shared_ptr<Expression>& expr);

  void add_to_memo(const std::shared_ptr<Expression>& expr,
		   const std::shared_ptr<Expression>& new_expr);
  bool has_memo(const std::shared_ptr<Expression>& expr);
  std::shared_ptr<Expression> get_memo(const std::shared_ptr<Expression>& expr);

  Problem::ptr parent_;

  // Keep track of expressions from original problem already inserted
  // to maintain the DAG.
  std::unordered_map<index_t, Expression::ptr> memo_;
};


} // namespace problem

} // namespace galini

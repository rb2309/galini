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
#include "detail.h"


#include <algorithm>
#include <vector>
#include <queue>

#include <iostream>

namespace galini {

namespace problem {

namespace detail {

std::vector<std::shared_ptr<Expression>>
collect_vertices(const std::shared_ptr<Expression>& root_expr) {
  std::queue<std::shared_ptr<Expression>> stack;
  std::vector<std::shared_ptr<Expression>> expressions;
  std::set<index_t> seen;

  // Do BFS visit on graph, accumulating expressions.
  stack.push(root_expr);

  while (stack.size() > 0) {
    auto current_expr = stack.front();
    stack.pop();
    auto already_visited = seen.find(current_expr->uid()) != seen.end();
    if (!already_visited) {
      expressions.push_back(current_expr);

      for (index_t i = 0; i < current_expr->num_children(); ++i) {
	seen.insert(current_expr->uid());
	stack.push(current_expr->nth_children(i));
      }
    }
  }

  std::reverse(expressions.begin(), expressions.end());

  return expressions;
}


std::shared_ptr<Expression> duplicate_tree(const std::shared_ptr<Expression>& expr,
					   std::unordered_map<index_t, std::shared_ptr<Expression>> existing_to_new) {
  auto vertices = collect_vertices(expr);
  std::cout << "DUplicate " << expr->uid() << std::endl;
  for (const auto& vertex : vertices) {
    auto num_children = vertex->num_children();
    std::cout << "  Vertex " << vertex->uid() << std::endl;
    std::vector<Expression::ptr> children(num_children);
    for (index_t i = 0; i < num_children; ++i) {
      auto existing_child = vertex->nth_children(i);
      std::cout << "    Child " << i << "  " << existing_child->uid() << std::endl;
      auto new_child = existing_to_new[existing_child->uid()];
      std::cout << "      New Child " << new_child->uid() << std::endl;
      children[i] = new_child;
    }
    auto new_vertex = vertex->duplicate(children);
    std::cout << "  New Vertex " << new_vertex->uid() << "  " << new_vertex->problem() << std::endl;
    existing_to_new[vertex->uid()] = new_vertex;
  }
  std::cout << "-> " << existing_to_new[expr->uid()]->uid() << std::endl;
  std::cout << "  problem = " << existing_to_new[expr->uid()]->problem() << "|" << std::endl;
  return existing_to_new[expr->uid()];
}


} // namespace detail

} // namespace problem

} // namespace galini

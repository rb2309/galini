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

#include <memory>

#include "expression/expression_base.h"

namespace galini {

namespace problem {

namespace detail {

using Expression = galini::expression::Expression;

std::vector<std::shared_ptr<Expression>>
collect_vertices(const std::shared_ptr<Expression>& root_expr);

std::shared_ptr<Expression>
duplicate_tree(const std::shared_ptr<Expression>& expr,
	       std::unordered_map<index_t, std::shared_ptr<Expression>> existing_to_new);

} // namespace detail

} // namespace problem

} // namespace galini

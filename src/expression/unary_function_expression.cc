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
#include "unary_function_expression.h"

#include "ad/ad.h"

namespace galini {

namespace expression {


#define EVAL_UNARY_FUNCTION_IMPL(class, func) \
  ADFloat class::eval(values_ptr<ADFloat>& values) const\
  { return func((*values)[child_]); }				\
  ADObject class::eval(values_ptr<ADObject>& values) const\
  { return func((*values)[child_]); }

#define DUPLICATE_FUNCTION_IMPL(class) \
  std::shared_ptr<Expression> \
  class::duplicate(const std::vector<typename Expression::ptr>& children) const \
  { assert(children.size() == 1); return std::make_shared<class>(problem(), children); }

EVAL_UNARY_FUNCTION_IMPL(AbsExpression, ad::abs)
EVAL_UNARY_FUNCTION_IMPL(SqrtExpression, ad::sqrt)
EVAL_UNARY_FUNCTION_IMPL(ExpExpression, ad::exp)
EVAL_UNARY_FUNCTION_IMPL(LogExpression, ad::log)
EVAL_UNARY_FUNCTION_IMPL(SinExpression, ad::sin)
EVAL_UNARY_FUNCTION_IMPL(CosExpression, ad::cos)
EVAL_UNARY_FUNCTION_IMPL(TanExpression, ad::tan)
EVAL_UNARY_FUNCTION_IMPL(AsinExpression, ad::asin)
EVAL_UNARY_FUNCTION_IMPL(AcosExpression, ad::acos)
EVAL_UNARY_FUNCTION_IMPL(AtanExpression, ad::atan)

DUPLICATE_FUNCTION_IMPL(AbsExpression)
DUPLICATE_FUNCTION_IMPL(SqrtExpression)
DUPLICATE_FUNCTION_IMPL(ExpExpression)
DUPLICATE_FUNCTION_IMPL(LogExpression)
DUPLICATE_FUNCTION_IMPL(SinExpression)
DUPLICATE_FUNCTION_IMPL(CosExpression)
DUPLICATE_FUNCTION_IMPL(TanExpression)
DUPLICATE_FUNCTION_IMPL(AsinExpression)
DUPLICATE_FUNCTION_IMPL(AcosExpression)
DUPLICATE_FUNCTION_IMPL(AtanExpression)

#undef EVAL_UNARY_FUNCTION_IMPL
#undef DUPLICATE_FUNCTION_IMPL

} // namespace expression

} // namespace galini

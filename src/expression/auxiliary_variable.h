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

#include "variable.h"

namespace galini {

namespace expression {


class AuxiliaryVariable : public Variable {
public:
  static const index_t DEFAULT_DEPTH = 1;
  using ptr = std::shared_ptr<AuxiliaryVariable>;

  AuxiliaryVariable(const std::shared_ptr<Problem>& problem,
		    const std::string& name,
		    py::object lower_bound,
		    py::object upper_bound,
		    py::object domain,
		    py::object reference);

  AuxiliaryVariable(const std::string& name,
		    py::object lower_bound,
		    py::object upper_bound,
		    py::object domain,
		    py::object reference);

  bool is_auxiliary() const override { return true; }
  py::object reference() const { return reference_; }
private:
  py::object reference_;
};

}  // namespace expression

} // namespace galini

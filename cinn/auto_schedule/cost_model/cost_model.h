// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _CINN__AUTO_SCHEDULE__COST_MODEL_
#define _CINN__AUTO_SCHEDULE__COST_MODEL_

#include <pybind11/embed.h>

#include <string>
#include <vector>

namespace cinn {
namespace auto_schedule {

/**
 * A C++ cost model which calls Python cost model via pybind
 *
 * Note: this class doesn't handle Python interpreter lifttime, users should
 * manage scoped_interpreter/initialize_interpreter/finalize_interpreter by
 * themselves. For pybind interpreter lifetime management, see:
 *
 *   https://pybind11.readthedocs.io/en/stable/advanced/embedding.html#interpreter-lifetime
 *   https://pybind11.readthedocs.io/en/stable/reference.html#_CPPv422initialize_interpreterbiPPCKcb
 */
class CostModel {
 public:
  CostModel();
  ~CostModel();

  void Train(const std::vector<std::vector<float>>& samples, const std::vector<float>& labels);

  std::vector<float> Predict(const std::vector<std::vector<float>>& samples);

  void Update(const std::vector<std::vector<float>>& samples, const std::vector<float>& labels);

  void Save(const std::string& path);

  void Load(const std::string& path);

 private:
  pybind11::object python_member_;
};

}  // namespace auto_schedule
}  // namespace cinn

#endif  // _CINN__AUTO_SCHEDULE__COST_MODEL_

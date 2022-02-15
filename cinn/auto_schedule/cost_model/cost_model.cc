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

#include "cinn/auto_schedule/cost_model/cost_model.h"

#include <pybind11/embed.h>

#include <string>
#include <vector>

#include "cinn/pybind/bind_utils.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::pybind::VectorToNumpy;

CostModel::CostModel() {
  pybind11::module cost_model_py_mod = pybind11::module::import("cinn.auto_schedule.cost_model");
  python_member_                     = cost_model_py_mod.attr("CostModel")();
}

CostModel::~CostModel() {
  // Do nothing, python_member_ will be destructed after CostModel destructor
}

void CostModel::Train(const std::vector<std::vector<float>>& samples, const std::vector<float>& labels) {
  pybind11::array np_samples = VectorToNumpy<float>(samples);
  pybind11::array np_labels  = VectorToNumpy<float>(labels);

  python_member_.attr("train")(np_samples, np_labels);
}

std::vector<float> CostModel::Predict(const std::vector<std::vector<float>>& samples) {
  pybind11::array np_samples = VectorToNumpy<float>(samples);

  pybind11::object py_result = python_member_.attr("predict")(np_samples);

  return py_result.cast<std::vector<float>>();
}

void CostModel::Update(const std::vector<std::vector<float>>& samples, const std::vector<float>& labels) {
  pybind11::array np_samples = VectorToNumpy<float>(samples);
  pybind11::array np_labels  = VectorToNumpy<float>(labels);

  python_member_.attr("update")(np_samples, np_labels);
}

void CostModel::Save(const std::string& path) { python_member_.attr("save")(py::str(path)); }

void CostModel::Load(const std::string& path) { python_member_.attr("load")(py::str(path)); }

}  // namespace auto_schedule
}  // namespace cinn

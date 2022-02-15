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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

//#include "cinn/pybind/bind_utils.h"

namespace cinn {
namespace auto_schedule {

// using ::cinn::pybind::VectorToNumpy;

// Convert 1D vector to py numpy
template <typename Dtype>
pybind11::array VectorToNumpy(const std::vector<Dtype>& vec) {
  return pybind11::array(pybind11::cast(vec));
}

// Convert 2D vector to py numpy
template <typename Dtype>
pybind11::array VectorToNumpy(const std::vector<std::vector<Dtype>>& vec) {
  if (vec.size() == 0) {
    return pybind11::array(pybind11::dtype::of<Dtype>(), {0, 0});
  }

  std::vector<size_t> shape{vec.size(), vec[0].size()};
  pybind11::array ret(pybind11::dtype::of<Dtype>(), shape);

  Dtype* py_data = static_cast<Dtype*>(ret.mutable_data());
  for (size_t i = 0; i < vec.size(); ++i) {
    assert(vec[i].size() == shape[1] && "Sub vectors must have same size in VectorToNumpy");
    const Dtype* sub_vec_data = vec[i].data();
    memcpy(py_data + (shape[0] * i), sub_vec_data, shape[0] * sizeof(Dtype));
  }
  return ret;
}

CostModel::CostModel() {
  pybind11::module sys_py_mod = pybind11::module::import("sys");
  // TODO fix the hard code here
  std::string site_pkg_str = "/usr/local/lib/python3.7/dist-packages";
  sys_py_mod.attr("path").attr("append")(site_pkg_str);
  auto path = sys_py_mod.attr("path").cast<std::vector<std::string>>();
  for (const std::string& s : path) {
    std::cout << s << std::endl;
  }
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

void CostModel::Save(const std::string& path) { python_member_.attr("save")(pybind11::str(path)); }

void CostModel::Load(const std::string& path) { python_member_.attr("load")(pybind11::str(path)); }

}  // namespace auto_schedule
}  // namespace cinn

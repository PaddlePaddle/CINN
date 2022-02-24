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
#include <mutex>
#include <string>
#include <vector>

namespace cinn {
namespace auto_schedule {

std::once_flag CostModel::init_once_flag_;

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
    memcpy(py_data + (shape[1] * i), vec[i].data(), shape[1] * sizeof(Dtype));
  }
  return ret;
}

void AddDistPkgToPythonSysPath() {
  pybind11::module sys_py_mod = pybind11::module::import("sys");
  // short version such as "3.7", "3.8", ...
  std::string py_short_version = sys_py_mod.attr("version").cast<std::string>().substr(0, 3);

  std::string site_pkg_str = "/usr/local/lib/python" + py_short_version + "/dist-packages";
  sys_py_mod.attr("path").attr("append")(site_pkg_str);

  std::string setuptools_str = site_pkg_str + "/setuptools-50.3.2-py3.7.egg";
  sys_py_mod.attr("path").attr("append")(setuptools_str);
}

CostModel::CostModel() {
  std::call_once(init_once_flag_, AddDistPkgToPythonSysPath);
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

  pybind11::array py_result = python_member_.attr("predict")(np_samples);
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

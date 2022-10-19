// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

TEST(Decomposer, relu) {
  NetBuilder builder("relu");
  auto x   = builder.CreateInput(Float(32), {20, 10}, "x");
  auto out = builder.Relu(x);

  auto relu_cpu = [](const std::vector<size_t>& lengths, const std::vector<void*>& ptrs) {
    size_t n   = lengths[0];
    float* x   = static_cast<float*>(ptrs[0]);
    float* out = static_cast<float*>(ptrs[1]);
    for (size_t i = 0; i < n; ++i) {
      float tmp_0 = x[i];
      out[i]      = tmp_0 > 0 ? tmp_0 : 0;
    }
  };

  std::vector<std::string> input_names        = {x.id().data()};
  std::vector<std::string> output_names       = {out->id};
  std::vector<std::vector<int>> output_shapes = {{20, 10}};
  RunAndCheck<float>(builder, input_names, output_names, output_shapes, relu_cpu, -1, 1);
}

TEST(Decomposer, relu_grad) {
  NetBuilder builder("relu_grad");
  auto dout = builder.CreateInput(Float(32), {20, 10}, "dout");
  auto out  = builder.CreateInput(Float(32), {20, 10}, "out");
  auto dx   = builder.ReluGrad(dout, out);

  auto relu_grad_cpu = [](const std::vector<size_t>& lengths, const std::vector<void*>& ptrs) {
    size_t n    = lengths[0];
    float* dout = static_cast<float*>(ptrs[0]);
    float* out  = static_cast<float*>(ptrs[1]);
    float* dx   = static_cast<float*>(ptrs[2]);
    for (size_t i = 0; i < n; ++i) {
      dx[i] = out[i] > 0 ? dout[i] : 0;
    }
  };

  std::vector<std::string> input_names        = {dout.id().data(), out.id().data()};
  std::vector<std::string> output_names       = {dx->id};
  std::vector<std::vector<int>> output_shapes = {{20, 10}};
  RunAndCheck<float>(builder, input_names, output_names, output_shapes, relu_grad_cpu, -1, 1);
}

}  // namespace cinn::frontend

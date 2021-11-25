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

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

void ComputeReluRef(const std::vector<float>& x, std::vector<float>* out, std::vector<bool>* mask) {
  size_t n = x.size();
  out->resize(n);
  mask->resize(n);

  for (size_t i = 0; i < n; ++i) {
    float tmp_0 = x[i];
    out->at(i)  = tmp_0 > 0 ? tmp_0 : 0;
    mask->at(i) = tmp_0 > 0;
  }
};

class ReluDecomposerTest : public DecomposerTest {
 public:
  void SetInputs(const std::vector<std::string>& input_names) { SetInputTensor<float>(input_names[0], &x_, -1, 1); }

  void CheckOutputs(const std::vector<std::string>& output_names) {
    GetOutputTensor<float>(output_names[0], &out_);
    // GetOutputTensor<bool>(output_names[1], &mask_);

    std::vector<float> out_ref;
    std::vector<bool> mask_ref;
    ComputeReluRef(x_, &out_ref, &mask_ref);

    CheckOutput<float>(out_, out_ref);
  }

  std::vector<float> x_;
  std::vector<float> out_;
  std::vector<bool> mask_;
};

TEST(Decomposer, relu) {
  NetBuilder builder("relu");
  auto x    = builder.CreateInput(Float(32), {20, 10}, "x");
  auto outs = builder.relu(x, true);

  ReluDecomposerTest relu_test;

  std::vector<std::string> input_names        = {x.id().data()};
  std::vector<std::string> output_names       = {outs[0]->id, outs[1]->id};
  std::vector<std::vector<int>> output_shapes = {{20, 10}, {20, 10}};
  relu_test.Execute(builder, input_names, output_names);
}

TEST(Decomposer, relu_grad) {
  NetBuilder builder("relu_grad");
  auto dout = builder.CreateInput(Float(32), {20, 10}, "dout");
  auto mask = builder.CreateInput(Bool(), {20, 10}, "mask");
  auto dx   = builder.relu_grad(dout, mask);

  auto relu_grad_cpu = [](const std::vector<size_t>& lengths, const std::vector<void*>& ptrs) {
    size_t n    = lengths[0];
    float* dout = static_cast<float*>(ptrs[0]);
    float* out  = static_cast<float*>(ptrs[1]);
    float* dx   = static_cast<float*>(ptrs[2]);
    for (size_t i = 0; i < n; ++i) {
      dx[i] = out[i] > 0 ? dout[i] : 0;
    }
  };

  std::vector<std::string> input_names        = {dout.id().data(), mask.id().data()};
  std::vector<std::string> output_names       = {dx->id};
  std::vector<std::vector<int>> output_shapes = {{20, 10}};
  RunAndCheck<float>(builder, input_names, output_names, output_shapes, relu_grad_cpu, -1, 1);
}

}  // namespace cinn::frontend

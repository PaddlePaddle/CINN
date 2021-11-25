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

void ComputeReluRef(const std::vector<float>& x,
                    std::vector<float>* out,
                    std::vector<bool>* mask = nullptr,
                    bool compute_mask       = false) {
  size_t n = x.size();

  out->resize(n);
  if (compute_mask) {
    mask->resize(n);
  }

  for (size_t i = 0; i < n; ++i) {
    float tmp_0 = x[i];
    out->at(i)  = tmp_0 > 0 ? tmp_0 : 0;
    if (compute_mask) {
      mask->at(i) = tmp_0 > 0;
    }
  }
};

void ComputeReluGradRef(const std::vector<float>& dout,
                        const std::vector<float> out,
                        const std::vector<bool> mask,
                        std::vector<float>* dx,
                        bool compute_mask = false) {
  size_t n = dout.size();
  if (compute_mask) {
    CHECK_EQ(mask.size(), n);
  } else {
    CHECK_EQ(out.size(), n);
  }

  dx->resize(n);
  for (size_t i = 0; i < n; ++i) {
    if (compute_mask) {
      dx->at(i) = mask[i] ? dout[i] : 0;
    } else {
      dx->at(i) = out[i] > 0 ? dout[i] : 0;
    }
  }
}

class ReluDecomposerTest : public DecomposerTest {
 public:
  void SetInputs(const std::vector<std::string>& input_names) { SetInputTensor<float>(input_names[0], &x_, -1, 1); }

  void CheckOutputs(const std::vector<std::string>& output_names, const std::vector<std::vector<int>>& output_shapes) {
    GetOutputTensor<float>(output_names[0], output_shapes[0], &out_);
    if (compute_mask_) {
      GetOutputTensor<bool>(output_names[1], output_shapes[1], &mask_);
    }

    std::vector<float> out_ref;
    std::vector<bool> mask_ref;
    ComputeReluRef(x_, &out_ref, &mask_ref, compute_mask_);

    LOG(INFO) << "output[out], var_name=" << output_names[0];
    CheckOutput<float>(out_, out_ref);
    if (compute_mask_) {
      LOG(INFO) << "output[mask], var_name=" << output_names[1];
      CheckOutput<bool>(mask_, mask_ref);
    }
  }

  bool compute_mask_{true};
  std::vector<float> x_;
  std::vector<float> out_;
  std::vector<bool> mask_;
};

class ReluGradDecomposerTest : public DecomposerTest {
 public:
  void SetInputs(const std::vector<std::string>& input_names) {
    SetInputTensor<float>(input_names[0], &dout_, -1, 1);
    if (compute_mask_) {
      SetInputTensor<bool>(input_names[1], &mask_, -10, 10);
    } else {
      SetInputTensor<float>(input_names[1], &out_, -1, 1);
    }
  }

  void CheckOutputs(const std::vector<std::string>& output_names, const std::vector<std::vector<int>>& output_shapes) {
    GetOutputTensor<float>(output_names[0], output_shapes[0], &dx_);

    std::vector<float> dx_ref;
    ComputeReluGradRef(dout_, out_, mask_, &dx_ref, compute_mask_);

    CheckOutput<float>(dx_, dx_ref);
  }

  bool compute_mask_{true};
  std::vector<float> dout_;
  std::vector<float> out_;
  std::vector<bool> mask_;
  std::vector<float> dx_;
};

TEST(Decomposer, relu) {
  ReluDecomposerTest relu_test;
  relu_test.compute_mask_ = true;
  std::vector<int> shape  = {10, 20};

  NetBuilder builder("relu");
  auto x    = builder.CreateInput(Float(32), shape, "x");
  auto outs = builder.relu(x, relu_test.compute_mask_);

  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names;
  std::vector<std::vector<int>> output_shapes;
  if (relu_test.compute_mask_) {
    output_names  = {outs[0]->id, outs[1]->id};
    output_shapes = {shape, shape};
  } else {
    output_names  = {outs[0]->id};
    output_shapes = {shape};
  }
  relu_test.Execute(builder, input_names, output_names, output_shapes);
}

TEST(Decomposer, relu_grad) {
  ReluGradDecomposerTest relu_grad_test;
  relu_grad_test.compute_mask_ = true;
  std::vector<int> shape       = {10, 20};

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::vector<int>> output_shapes = {shape};

  NetBuilder builder("relu_grad");
  if (relu_grad_test.compute_mask_) {
    auto dout    = builder.CreateInput(Float(32), shape, "dout");
    auto mask    = builder.CreateInput(Bool(), shape, "mask");
    auto dx      = builder.relu_grad(dout, mask);
    input_names  = {dout.id().data(), mask.id().data()};
    output_names = {dx->id};
  } else {
    auto dout    = builder.CreateInput(Float(32), shape, "dout");
    auto out     = builder.CreateInput(Float(32), shape, "out");
    auto dx      = builder.relu_grad(dout, out);
    input_names  = {dout.id().data(), out.id().data()};
    output_names = {dx->id};
  }
  relu_grad_test.Execute(builder, input_names, output_names, output_shapes);
}

}  // namespace cinn::frontend

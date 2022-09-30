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

#pragma once

#include "cinn/auto_schedule/tests/test_program_builder.h"
#include "cinn/frontend/net_builder.h"

namespace cinn {
namespace auto_schedule {

class AddProgramBuilder : public TestProgramBuilder {
 public:
  AddProgramBuilder(const std::vector<int32_t>& input_shape_x, const std::vector<int32_t>& input_shape_y)
      : input_shape_x_(input_shape_x), input_shape_y_(input_shape_y) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("add_net_builder");
    auto x = builder.CreateInput(Float(32), input_shape_x_, "X");
    auto y = builder.CreateInput(Float(32), input_shape_y_, "Y");

    auto mul_out = builder.Add(x, y);
    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_x_;
  std::vector<int32_t> input_shape_y_;
};

class MulProgramBuilder : public TestProgramBuilder {
 public:
  MulProgramBuilder(const std::vector<int32_t>& input_shape_x,
                    const std::vector<int32_t>& input_shape_y,
                    int x_num_col_dims = 1,
                    int y_num_col_dims = 1)
      : input_shape_x_(input_shape_x),
        input_shape_y_(input_shape_y),
        x_num_col_dims_(x_num_col_dims),
        y_num_col_dims_(y_num_col_dims) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("mul_net_builder");
    auto x = builder.CreateInput(Float(32), input_shape_x_, "X");
    auto y = builder.CreateInput(Float(32), input_shape_y_, "Y");

    auto mul_out = builder.Mul(x, y, x_num_col_dims_, y_num_col_dims_);
    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_x_;
  std::vector<int32_t> input_shape_y_;
  int x_num_col_dims_;
  int y_num_col_dims_;
};

class MatmulProgramBuilder : public TestProgramBuilder {
 public:
  MatmulProgramBuilder(const std::vector<int32_t>& input_shape_x,
                       const std::vector<int32_t>& input_shape_y,
                       bool trans_x = false,
                       bool trans_y = false,
                       float alpha  = 1.0f)
      : input_shape_x_(input_shape_x),
        input_shape_y_(input_shape_y),
        trans_x_(trans_x),
        trans_y_(trans_y),
        alpha_(alpha) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("matmul_net_builder");
    auto x = builder.CreateInput(Float(32), input_shape_x_, "X");
    auto y = builder.CreateInput(Float(32), input_shape_y_, "Y");

    auto mul_out = builder.Matmul(x, y, trans_x_, trans_y_, alpha_);
    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_x_;
  std::vector<int32_t> input_shape_y_;
  bool trans_x_;
  bool trans_y_;
  float alpha_;
};

class ReluProgramBuilder : public TestProgramBuilder {
 public:
  ReluProgramBuilder(std::vector<int32_t> input_shape) : input_shape_(input_shape) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("relu_net_builder");
    auto x = builder.CreateInput(Float(32), input_shape_, "X");
    auto y = builder.Relu(x);

    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_;
};

class Conv2dProgramBuilder : public TestProgramBuilder {
 public:
  Conv2dProgramBuilder(const std::vector<int32_t>& input_shape,
                       const std::vector<int32_t>& weight_shape,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::vector<int>& dilations,
                       int groups                           = 1,
                       const std::string& data_format       = "NCHW",
                       const std::string& padding_algorithm = "EXPLICIT")
      : input_shape_(input_shape),
        weight_shape_(weight_shape),
        strides_(strides),
        paddings_(paddings),
        dilations_(dilations),
        groups_(groups),
        data_format_(data_format),
        padding_algorithm_(padding_algorithm_) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("conv2d_net_builder");
    auto x       = builder.CreateInput(Float(32), input_shape_, "X");
    auto weights = builder.CreateInput(Float(32), weight_shape_, "W");
    auto out = builder.Conv2d(x, weights, strides_, paddings_, dilations_, groups_, data_format_, padding_algorithm_);

    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_;
  std::vector<int32_t> weight_shape_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int groups_;
  std::string data_format_;
  std::string padding_algorithm_;
};

class Pool2dProgramBuilder : public TestProgramBuilder {
 public:
  Pool2dProgramBuilder(const std::vector<int32_t>& input_shape,
                       const std::string& pooling_type,
                       const std::vector<int>& ksize,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       bool ceil_mode                       = false,
                       bool exclusive                       = true,
                       bool global_pooling                  = false,
                       const std::string& data_format       = "NCHW",
                       bool adaptive                        = false,
                       const std::string& padding_algorithm = "EXPLICIT")
      : input_shape_(input_shape),
        pooling_type_(pooling_type),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        ceil_mode_(ceil_mode),
        exclusive_(exclusive),
        global_pooling_(global_pooling),
        data_format_(data_format),
        adaptive_(adaptive),
        padding_algorithm_(padding_algorithm) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("pool2d_net_builder");
    auto x   = builder.CreateInput(Float(32), input_shape_, "X");
    auto out = builder.Pool2d(x,
                              pooling_type_,
                              ksize_,
                              strides_,
                              paddings_,
                              ceil_mode_,
                              exclusive_,
                              global_pooling_,
                              data_format_,
                              adaptive_,
                              padding_algorithm_);

    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_;
  std::string pooling_type_;
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool ceil_mode_;
  bool exclusive_;
  bool global_pooling_;
  std::string data_format_;
  bool adaptive_;
  std::string padding_algorithm_;
};

class BatchNormProgramBuilder : public TestProgramBuilder {
 public:
  BatchNormProgramBuilder(const std::vector<int32_t>& input_shape,
                          const std::vector<int32_t>& scale_shape,
                          const std::vector<int32_t>& bias_shape,
                          const std::vector<int32_t>& mean_shape,
                          const std::vector<int32_t>& variance_shape,
                          float epsilon                  = 1e-5f,
                          float momentum                 = 0.9f,
                          const std::string& data_layout = "NCHW",
                          bool is_test                   = false)
      : input_shape_(input_shape),
        scale_shape_(scale_shape),
        bias_shape_(bias_shape),
        mean_shape_(mean_shape),
        variance_shape_(variance_shape),
        epsilon_(epsilon),
        momentum_(momentum),
        data_layout_(data_layout),
        is_test_(is_test) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("pool2d_net_builder");
    auto x        = builder.CreateInput(Float(32), input_shape_, "X");
    auto scale    = builder.CreateInput(Float(32), scale_shape_, "scale");
    auto bias     = builder.CreateInput(Float(32), bias_shape_, "bias");
    auto mean     = builder.CreateInput(Float(32), mean_shape_, "mean");
    auto variance = builder.CreateInput(Float(32), variance_shape_, "variance");
    auto out      = builder.BatchNorm(x, scale, bias, mean, variance, epsilon_, momentum_, data_layout_, is_test_);

    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_;
  std::vector<int32_t> scale_shape_;
  std::vector<int32_t> bias_shape_;
  std::vector<int32_t> mean_shape_;
  std::vector<int32_t> variance_shape_;
  float epsilon_;
  float momentum_;
  std::string data_layout_;
  bool is_test_;
};

class ReshapeProgramBuilder : public TestProgramBuilder {
 public:
  ReshapeProgramBuilder(const std::vector<int32_t>& input_shape, const std::vector<int32_t>& output_shape)
      : input_shape_(input_shape), output_shape_(output_shape) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("reshape_net_builder");
    auto x = builder.CreateInput(Float(32), input_shape_, "X");
    auto y = builder.Reshape(x, output_shape_);

    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_;
  std::vector<int32_t> output_shape_;
};

class SoftmaxProgramBuilder : public TestProgramBuilder {
 public:
  SoftmaxProgramBuilder(const std::vector<int32_t>& input_shape,
                        int axis                       = -1,
                        const std::string& data_format = "AnyLayout")
      : input_shape_(input_shape), axis_(axis), data_format_(data_format) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("softmax_net_builder");
    auto x = builder.CreateInput(Float(32), input_shape_, "X");
    auto y = builder.Softmax(x, axis_, data_format_);

    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_;
  int axis_;
  std::string data_format_;
};

class ScaleProgramBuilder : public TestProgramBuilder {
 public:
  ScaleProgramBuilder(const std::vector<int32_t>& input_shape,
                      float scale           = 1.0f,
                      float bias            = 0.0f,
                      bool bias_after_scale = true)
      : input_shape_(input_shape), scale_(scale), bias_(bias), bias_after_scale_(bias_after_scale) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("scale_net_builder");
    auto x = builder.CreateInput(Float(32), input_shape_, "X");
    auto y = builder.Scale(x, scale_, bias_, bias_after_scale_);

    return builder.Build();
  }

 private:
  std::vector<int32_t> input_shape_;
  float scale_;
  float bias_;
  bool bias_after_scale_;
};

}  // namespace auto_schedule
}  // namespace cinn

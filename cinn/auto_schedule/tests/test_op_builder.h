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

namespace cinn {
namespace auto_schedule {

class AddOpBuilder : public TestOpBuilder {
 public:
  AddOpBuilder(const std::vector<int32_t>& input_shape_x, const std::vector<int32_t>& input_shape_y)
      : TestOpBuilder("add_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape_x, "X");
    auto y = builder_.CreateInput(Float(32), input_shape_y, "Y");
    builder_.Add(x, y);
  }
};

class MulOpBuilder : public TestOpBuilder {
 public:
  MulOpBuilder(const std::vector<int32_t>& input_shape_x,
               const std::vector<int32_t>& input_shape_y,
               int x_num_col_dims = 1,
               int y_num_col_dims = 1)
      : TestOpBuilder("mul_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape_x, "X");
    auto y = builder_.CreateInput(Float(32), input_shape_y, "Y");
    builder_.Mul(x, y, x_num_col_dims, y_num_col_dims, true);
  }
};

class MatmulOpBuilder : public TestOpBuilder {
 public:
  MatmulOpBuilder(const std::vector<int32_t>& input_shape_x,
                  const std::vector<int32_t>& input_shape_y,
                  bool trans_x = false,
                  bool trans_y = false,
                  float alpha  = 1.0f)
      : TestOpBuilder("matmul_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape_x, "X");
    auto y = builder_.CreateInput(Float(32), input_shape_y, "Y");
    builder_.Matmul(x, y, trans_x, trans_y, alpha);
  }
};

class ReluOpBuilder : public TestOpBuilder {
 public:
  ReluOpBuilder(std::vector<int32_t> input_shape) : TestOpBuilder("relu_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape, "X");
    builder_.Relu(x);
  }
};

class Conv2dOpBuilder : public TestOpBuilder {
 public:
  Conv2dOpBuilder(const std::vector<int32_t>& input_shape,
                  const std::vector<int32_t>& weight_shape,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  int groups                           = 1,
                  const std::string& data_format       = "NCHW",
                  const std::string& padding_algorithm = "EXPLICIT")
      : TestOpBuilder("conv2d_net_builder") {
    auto x       = builder_.CreateInput(Float(32), input_shape, "X");
    auto weights = builder_.CreateInput(Float(32), weight_shape, "W");
    builder_.Conv2d(x, weights, strides, paddings, dilations, groups, data_format, padding_algorithm);
  }
};

class Pool2dOpBuilder : public TestOpBuilder {
 public:
  Pool2dOpBuilder(const std::vector<int32_t>& input_shape,
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
      : TestOpBuilder("pool2d_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape, "X");
    builder_.Pool2d(x,
                    pooling_type,
                    ksize,
                    strides,
                    paddings,
                    ceil_mode,
                    exclusive,
                    global_pooling,
                    data_format,
                    adaptive,
                    padding_algorithm);
  }
};

class BatchNormOpBuilder : public TestOpBuilder {
 public:
  BatchNormOpBuilder(const std::vector<int32_t>& input_shape,
                     const std::vector<int32_t>& scale_shape,
                     const std::vector<int32_t>& bias_shape,
                     const std::vector<int32_t>& mean_shape,
                     const std::vector<int32_t>& variance_shape,
                     float epsilon                  = 1e-5f,
                     float momentum                 = 0.9f,
                     const std::string& data_layout = "NCHW",
                     bool is_test                   = false)
      : TestOpBuilder("pool2d_net_builder") {
    auto x        = builder_.CreateInput(Float(32), input_shape, "X");
    auto scale    = builder_.CreateInput(Float(32), scale_shape, "scale");
    auto bias     = builder_.CreateInput(Float(32), bias_shape, "bias");
    auto mean     = builder_.CreateInput(Float(32), mean_shape, "mean");
    auto variance = builder_.CreateInput(Float(32), variance_shape, "variance");
    builder_.BatchNorm(x, scale, bias, mean, variance, epsilon, momentum, data_layout, is_test);
  }
};

class ReshapeOpBuilder : public TestOpBuilder {
 public:
  ReshapeOpBuilder(const std::vector<int32_t>& input_shape, const std::vector<int32_t>& output_shape)
      : TestOpBuilder("reshape_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape, "X");
    builder_.Reshape(x, output_shape);
  }
};

class SoftmaxOpBuilder : public TestOpBuilder {
 public:
  SoftmaxOpBuilder(const std::vector<int32_t>& input_shape, int axis = -1, const std::string& data_format = "AnyLayout")
      : TestOpBuilder("softmax_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape, "X");
    builder_.Softmax(x, {axis});
  }
};

class ScaleOpBuilder : public TestOpBuilder {
 public:
  ScaleOpBuilder(const std::vector<int32_t>& input_shape,
                 float scale           = 1.0f,
                 float bias            = 0.0f,
                 bool bias_after_scale = true)
      : TestOpBuilder("scale_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape, "X");
    builder_.Scale(x, scale, bias, bias_after_scale);
  }
};

class LookupTableOpBuilder : public TestOpBuilder {
 public:
  LookupTableOpBuilder(const std::vector<int32_t>& table_shape,
                       const std::vector<int32_t>& ids_shape,
                       int64_t padding_idx)
      : TestOpBuilder("lookup_net_builder") {
    auto t = builder_.CreateInput(Float(32), table_shape, "table");
    auto i = builder_.CreateInput(Int(64), ids_shape, "ids");
    builder_.LookupTable(t, i, padding_idx);
  }
};

class GatherOpBuilder : public TestOpBuilder {
 public:
  GatherOpBuilder(const std::vector<int32_t>& operand_shape, const std::vector<int32_t>& index_shape, int32_t axis)
      : TestOpBuilder("gather_builder") {
    auto operand = builder_.CreateInput(Float(32), operand_shape, "operand");
    auto index   = builder_.CreateInput(Int(32), index_shape, "index");
    builder_.Gather(operand, index, axis);
  }
};

class ReduceSumOpBuilder : public TestOpBuilder {
 public:
  ReduceSumOpBuilder(const std::vector<int32_t>& input_shape, const std::vector<int32_t>& reduce_dim)
      : TestOpBuilder("reduce_sum_net_builder") {
    auto x = builder_.CreateInput(Float(32), input_shape, "X");
    builder_.ReduceSum(x, reduce_dim);
  }
};

}  // namespace auto_schedule
}  // namespace cinn

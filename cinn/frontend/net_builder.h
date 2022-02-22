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

#pragma once

#include <string>
#include <vector>

#include "cinn/frontend/base_builder.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

// clang-format off
#define NETBUILDER_UNARY_OP_FOREACH(macro__)    \
    macro__(Sqrt)                               \
    macro__(Tanh)                               \
    macro__(Relu)                               \
    macro__(Sigmoid)                            \
    macro__(Identity)

#define NETBUILDER_BINARY_OP_FOREACH(macro__)   \
    macro__(Sub)                                \
    macro__(Div)                                \
    macro__(Matmul)                             \
    macro__(ReluGrad)
// clang-format on

class NetBuilder : public BaseBuilder {
 public:
  using BaseBuilder::BaseBuilder;

#define NETBUILDER_UNARY_OP_DECL(func_name__) Variable func_name__(const Variable& operand);
  NETBUILDER_UNARY_OP_FOREACH(NETBUILDER_UNARY_OP_DECL)
#undef NETBUILDER_UNARY_OP_DECL

#define NETBUILDER_BINARY_OP_DECL(func_name__) Variable func_name__(const Variable& lhs, const Variable& rhs);
  NETBUILDER_BINARY_OP_FOREACH(NETBUILDER_BINARY_OP_DECL)
#undef NETBUILDER_BINARY_OP_DECL

  /**
   * Multiply two matrix.
   */
  Variable Mul(const Variable& a, const Variable& b, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * Multiply two matrix and add a bias.
   */
  Variable MulBias(
      const Variable& a, const Variable& b, const Variable& c, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * Add two matrix(with broadcast).
   */
  Variable Add(const Variable& a, const Variable& b);

  /**
   * Add two tensors element-wise.
   */
  Variable ElementwiseAdd(const Variable& a, const Variable& b, int axis = -1);

  /**
   * The gradient of elementwise_add.
   */
  const std::vector<Variable>& ElementwiseAddGrad(const Variable& dout,
                                                  const Variable& x,
                                                  const Variable& y,
                                                  int axis = -1);

  /**
   * Multiply two tensors element-wise.
   */
  Variable ElementwiseMul(const Variable& a, const Variable& b, int axis = -1);

  Variable Relu6(const Variable& a, float threshold = 6.0f);

  /**
   * Compute the sum of Variable x along the given dim.
   */
  Variable ReduceSum(const Variable& x, const std::vector<int>& dim, bool keep_dim = false);

  /**
   * The convolution2D layer calculates the output based on the input, filter
   * and strides, paddings, dilations, groups parameters.
   */
  Variable Conv2d(const Variable& a,
                  const Variable& b,
                  const std::vector<int>& strides      = {1, 1},
                  const std::vector<int>& paddings     = {0, 0},
                  const std::vector<int>& dilations    = {1, 1},
                  int groups                           = 1,
                  const std::string& data_format       = "NCHW",
                  const std::string& padding_algorithm = "EXPLICIT");

  Variable DepthwiseConv2d(const Variable& a,
                           const Variable& b,
                           const std::vector<int>& strides      = {1, 1},
                           const std::vector<int>& paddings     = {0, 0},
                           const std::vector<int>& dilations    = {1, 1},
                           int groups                           = 1,
                           const std::string& data_format       = "NCHW",
                           const std::string& padding_algorithm = "EXPLICIT");

  Variable Pool2d(const Variable& a,
                  const std::string& pooling_type,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides      = {1, 1},
                  const std::vector<int>& paddings     = {0, 0},
                  bool ceil_mode                       = false,
                  bool exclusive                       = true,
                  bool global_pooling                  = false,
                  const std::string& data_format       = "NCHW",
                  bool adaptive                        = false,
                  const std::string& padding_algorithm = "EXPLICIT");

  /**
   * The batchnorm layer can be used as a normalizer function
   * for convolution or fully_connected operations.
   * is_test(true): batch norm infer (default), output={y}
   * is_test(false): batch norm training, outputs={y, saved_mean, saved_variance, moving_mean, moving_variance}
   */
  std::vector<Variable> BatchNorm(const Variable& a,
                                  const Variable& scale,
                                  const Variable& bias,
                                  const Variable& mean,
                                  const Variable& variance,
                                  float epsilon                  = 1e-5f,
                                  float momentum                 = 0.9f,
                                  const std::string& data_layout = "NCHW",
                                  bool is_test                   = false);

  // batch norm grad, output(x_grad, scale_grad, bias_grad)
  std::vector<Variable> BatchNormGrad(const Variable& dy,
                                      const Variable& x,
                                      const Variable& scale,
                                      const Variable& save_mean,
                                      const Variable& save_variance,
                                      const float epsilon            = 1e-5,
                                      const std::string& data_layout = "NCHW");

  Variable Scale(const Variable& a, float scale = 1.0f, float bias = 0.0f, bool bias_after_scale = true);

  Variable Softmax(const Variable& a, int axis = -1, const std::string& data_format = "AnyLayout");

  Variable DropoutInfer(const Variable& a,
                        float dropout_prob                        = 0.5f,
                        const std::string& dropout_implementation = "downgrade_in_infer");

  Variable Sum(const std::vector<Variable>& inputs);

  // conv2d grad, output(grad_x, grad_w)
  std::vector<Variable> Conv2dGrad(const Variable& dy,
                                   const Variable& x,
                                   const Variable& w,
                                   const std::vector<int>& strides      = {1, 1},
                                   const std::vector<int>& paddings     = {0, 0},
                                   const std::vector<int>& dilations    = {1, 1},
                                   const int groups                     = 1,
                                   const std::string& data_format       = "NCHW",
                                   const std::string& padding_algorithm = "EXPLICIT");

  template <typename T>
  Variable FillConstant(const std::vector<int>& shape, float value, const std::string& name, bool force_cpu = false) {
    Instruction instr("fill_constant");
    instr.SetInputs({});
    instr.SetAttr("shape", shape);
    instr.SetAttr("value", value);
    instr.SetAttr("force_cpu", force_cpu);

    InferShape(instr);
    AppendInstruction(instr);
    auto out = instr.GetOutput(0);
    out.set_id(name);
    return out;
  }
};

}  // namespace frontend
}  // namespace cinn

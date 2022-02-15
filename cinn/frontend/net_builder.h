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
    macro__(sqrt)                               \
    macro__(tanh)                               \
    macro__(relu)                               \
    macro__(sigmoid)                            \
    macro__(identity)

#define NETBUILDER_BINARY_OP_FOREACH(macro__)   \
    macro__(sub)                                \
    macro__(div)                                \
    macro__(matmul)                             \
    macro__(relu_grad)
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
   * Add two variables.
   */
  Variable add(const Variable& a, const Variable& b);

  /**
   * Reshape Variable.
   */
  Variable reshape(const Variable& a, const std::vector<int>& shape);

  /**
   * Transpose matrix.
   */
  Variable transpose(const Variable& a, const std::vector<int>& axis);

  /**
   * Concat tensors among the axis dimension.
   */
  Variable concat(const std::vector<Variable>& inputs, int axis = 0);

  /**
   * Multiply two matrix.
   */
  Variable mul(const Variable& a, const Variable& b, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * Multiply two matrix and add a bias.
   */
  Variable mulbias(
      const Variable& a, const Variable& b, const Variable& c, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * Add two tensors element-wise.
   */
  Variable elementwise_add(const Variable& a, const Variable& b, int axis = -1);

  /**
   * The gradient of elementwise_add.
   */
  const std::vector<Variable>& elementwise_add_grad(const Variable& dout,
                                                    const Variable& x,
                                                    const Variable& y,
                                                    int axis = -1);

  /**
   * Multiply two tensors element-wise.
   */
  Variable elementwise_mul(const Variable& a, const Variable& b, int axis = -1);

  Variable relu6(const Variable& a, float threshold = 6.0f);

  /**
   * This API reverses the Variable x along the given axis.
   * Example 1: x = [[0, 1], [2, 3], [4, 5]], axis = [0]
   *            output = [[4, 5], [2, 3], [0, 1]]
   * Example 2: x = [[0, 1], [2, 3], [4, 5]], axis = [0, 1]
   *            output = [[5, 4], [3, 2], [1, 0]]
   */
  Variable reverse(const Variable& x, const std::vector<int>& axis);

  /**
   * Compute the sum of Variable x along the given dim.
   */
  Variable reduce_sum(const Variable& x, const std::vector<int>& dim, bool keep_dim = false);

  /**
   * The convolution2D layer calculates the output based on the input, filter
   * and strides, paddings, dilations, groups parameters.
   */
  Variable conv2d(const Variable& a,
                  const Variable& b,
                  const std::vector<int>& strides      = {1, 1},
                  const std::vector<int>& paddings     = {0, 0},
                  const std::vector<int>& dilations    = {1, 1},
                  int groups                           = 1,
                  const std::string& data_format       = "NCHW",
                  const std::string& padding_algorithm = "EXPLICIT");

  Variable depthwise_conv2d(const Variable& a,
                            const Variable& b,
                            const std::vector<int>& strides      = {1, 1},
                            const std::vector<int>& paddings     = {0, 0},
                            const std::vector<int>& dilations    = {1, 1},
                            int groups                           = 1,
                            const std::string& data_format       = "NCHW",
                            const std::string& padding_algorithm = "EXPLICIT");

  Variable pool2d(const Variable& a,
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
  std::vector<Variable> batchnorm(const Variable& a,
                                  const Variable& scale,
                                  const Variable& bias,
                                  const Variable& mean,
                                  const Variable& variance,
                                  float epsilon                  = 1e-5f,
                                  float momentum                 = 0.9f,
                                  const std::string& data_layout = "NCHW",
                                  bool is_test                   = false);

  // batch norm grad, output(x_grad, scale_grad, bias_grad)
  std::vector<Variable> batch_norm_grad(const Variable& dy,
                                        const Variable& x,
                                        const Variable& scale,
                                        const Variable& save_mean,
                                        const Variable& save_variance,
                                        const float epsilon            = 1e-5,
                                        const std::string& data_layout = "NCHW");

  Variable scale(const Variable& a, float scale = 1.0f, float bias = 0.0f, bool bias_after_scale = true);

  Variable softmax(const Variable& a, int axis = -1, const std::string& data_format = "AnyLayout");

  Variable slice(const Variable& a,
                 const std::vector<int>& axes,
                 const std::vector<int>& starts        = {},
                 const std::vector<int>& ends          = {},
                 const std::vector<int>& infer_flags   = {},
                 const std::vector<int>& decrease_axis = {});

  Variable dropout_infer(const Variable& a,
                         float dropout_prob                        = 0.5f,
                         const std::string& dropout_implementation = "downgrade_in_infer");

  Variable sum(const std::vector<Variable>& inputs);

  // conv2d grad, output(grad_x, grad_w)
  std::vector<Variable> conv2d_grad(const Variable& dy,
                                    const Variable& x,
                                    const Variable& w,
                                    const std::vector<int>& strides      = {1, 1},
                                    const std::vector<int>& paddings     = {0, 0},
                                    const std::vector<int>& dilations    = {1, 1},
                                    const int groups                     = 1,
                                    const std::string& data_format       = "NCHW",
                                    const std::string& padding_algorithm = "EXPLICIT");

  /**
   * Reduce tensor by the kind operator among the dim.
   */
  Variable reduce(const Variable& a,
                  ReduceKind kind             = ReduceKind::kSum,
                  const std::vector<int>& dim = std::vector<int>{},
                  bool keep_dim               = false);

  Variable broadcast_to(const Variable& a, const std::vector<int>& out_shape, const std::vector<int>& broadcast_axes);

 private:
  Variable UnaryOp(const std::string& op_type, const Variable& operand);

  Variable BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs);
};

}  // namespace frontend
}  // namespace cinn

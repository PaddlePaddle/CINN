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

class NetBuilder : public BaseBuilder {
 public:
  using BaseBuilder::BaseBuilder;

  /**
   * Add two variables.
   */
  Variable add(const Variable& a, const Variable& b);

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

  /**
   * Apply Rectified Linear Unit on input Variable.
   * Actually apply: outupt = max(input,0)
   */
  Variable relu(const Variable& a);

  /**
   * The gradient of Rectified Linear Unit.
   * Actually apply: dx = dout * (out > 0)
   */
  Variable relu_grad(const Variable& dout, const Variable& out);

  Variable relu6(const Variable& a, float threshold = 6.0f);

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
   */
  Variable batchnorm(const Variable& a,
                     const Variable& scale,
                     const Variable& bias,
                     const Variable& mean,
                     const Variable& variance,
                     float epsilon                  = 1e-5f,
                     float momentum                 = 0.9f,
                     const std::string& data_layout = "NCHW");

  Variable scale(const Variable& a, float scale = 1.0f, float bias = 0.0f, bool bias_after_scale = true);

  Variable softmax(const Variable& a, int axis = -1, const std::string& data_format = "AnyLayout");

  Variable sigmoid(const Variable& a);

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

  // batch norm grad, output(grad_x, grad_scale, grad_bias)
  std::vector<Variable> batch_norm_grad(const Variable& x,
                                        const Variable& dy,
                                        const Variable& scale,
                                        const Variable& save_mean,
                                        const Variable& save_var,
                                        const float epsilon            = 1e-6,
                                        const std::string& data_layout = "NCHW");
};

}  // namespace frontend
}  // namespace cinn

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

#include "cinn/frontend/decomposer_registry.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace decomposer {

struct BatchNormHelper {
  BatchNormHelper(CinnBuilder* cinn_builder,
                  const std::vector<int>& arg_x_shape,
                  const std::vector<int>& arg_param_shape,
                  std::string data_layout) {
    CHECK_EQ(arg_x_shape.size(), 4UL) << "Only 4-D input tensor is supported, but get " << arg_x_shape.size()
                                      << "-D input tensor.";

    builder     = cinn_builder;
    x_shape     = arg_x_shape;
    param_shape = arg_param_shape;

    if (data_layout == "NCHW") {
      channel_dim   = 1;
      reduce_dim    = {0, 2, 3};
      element_count = x_shape[0] * x_shape[2] * x_shape[3];
    } else if (data_layout == "NHWC") {
      channel_dim   = 3;
      reduce_dim    = {0, 1, 2};
      element_count = x_shape[0] * x_shape[1] * x_shape[2];
    } else {
      LOG(FATAL) << data_layout << " setting is not support!";
    }
  }

  template <typename T>
  Variable GetTensorFromScalar(T value, std::string name, const std::vector<int>& shape) {
    return builder->BroadcastTo(builder->ConstScalar<T>(value, common::UniqName(name)), shape, {0});
  }

  Variable Mean(Variable x, Variable element_count_1d) {
    auto sum  = builder->Reduce(x, ReduceKind::kSum, reduce_dim);
    auto mean = builder->Div(sum, element_count_1d);
    return mean;
  }

  Variable Variance(Variable x, Variable mean, Variable element_count_1d) {
    auto x_square      = builder->Mul(x, builder->Identity(x));
    auto x_square_sum  = builder->Reduce(x_square, ReduceKind::kSum, reduce_dim);
    auto x_square_mean = builder->Div(x_square_sum, element_count_1d);
    auto variance      = builder->Sub(x_square_mean, builder->Mul(mean, builder->Identity(mean)));
    return variance;
  }

  Variable StdVarianceInv4d(Variable variance, float epsilon) {
    // auto epsilon_1d          = GetTensorFromScalar<float>(epsilon, "epsilon", param_shape);
    // auto std_variance_inv    = builder->Rsqrt(builder->Add(variance, epsilon_1d));
    // auto std_variance_inv_4d = builder->BroadcastTo(std_variance_inv, x_shape, {channel_dim});
    auto epsilon_4d          = GetTensorFromScalar<float>(epsilon, "epsilon", x_shape);
    auto variance_4d         = builder->BroadcastTo(variance, x_shape, {channel_dim});
    auto std_variance_inv_4d = builder->Rsqrt(builder->Add(variance_4d, epsilon_4d));
    return std_variance_inv_4d;
  }

  CinnBuilder* builder{nullptr};
  std::vector<int> x_shape;
  std::vector<int> param_shape;
  std::vector<int> reduce_dim;
  float element_count{0};
  int channel_dim{0};
};

void batch_norm_train(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 5UL) << "The number of the given inputs is not equal to the required for op "
                                      << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 5UL) << "The number of the given outputs is not equal to the required for op "
                                       << instr->op_type;

  auto& x               = instr->inputs[0];
  auto& scale           = instr->inputs[1];
  auto& bias            = instr->inputs[2];
  auto& moving_mean     = instr->inputs[3];
  auto& moving_variance = instr->inputs[4];

  float epsilon      = instr.GetAttrs<float>("epsilon");
  float momentum     = instr.GetAttrs<float>("momentum");
  std::string layout = instr.GetAttrs<std::string>("data_layout");

  CinnBuilder* builder = context.builder();
  BatchNormHelper helper(builder, x->shape, scale->shape, layout);

  auto element_count_1d = helper.GetTensorFromScalar<float>(helper.element_count, "element_count", scale->shape);

  // mean = reduce_sum(x) / nhw, shape = [c]
  auto mean    = helper.Mean(x, element_count_1d);
  auto mean_4d = builder->BroadcastTo(mean, x->shape, {helper.channel_dim});

  // variance = reduce_sum(x * x) / nhw - mean * mean, shape = [c], simplified by equation: E(x^2) - [E(x)]^2
  auto variance = helper.Variance(x, mean, element_count_1d);

  // std_variance_inv = rsqrt(variance + epsilon), shape = [c]
  auto std_variance_inv_4d = helper.StdVarianceInv4d(variance, epsilon);

  // y = scale * (x - mean) * std_variance_inv + bias, shape = [n, c, h, w]
  auto scale_4d          = builder->BroadcastTo(scale, x->shape, {helper.channel_dim});
  auto bias_4d           = builder->BroadcastTo(bias, x->shape, {helper.channel_dim});
  auto normalized        = builder->Mul(builder->Sub(x, mean_4d), std_variance_inv_4d);
  auto scaled_normalized = builder->Mul(normalized, scale_4d);
  auto y                 = builder->Add(scaled_normalized, bias_4d);

  auto factor_0 = helper.GetTensorFromScalar<float>(momentum, "factor_0", moving_mean->shape);
  auto factor_1 = helper.GetTensorFromScalar<float>(1.0f - momentum, "factor_1", moving_mean->shape);

  // moving_mean = moving_mean * momentum + (1.0 - momentum) * mean, shape = [c]
  auto new_moving_mean = builder->Add(builder->Mul(moving_mean, factor_0), builder->Mul(mean, factor_1));

  // moving_variance = moving_variance * momentum + (1.0 - momentum) * variance, shape = [c]
  auto new_moving_variance = builder->Add(builder->Mul(moving_variance, factor_0), builder->Mul(variance, factor_1));

  context.MapOutToOrigin(y, instr->outputs[0]);
  context.MapOutToOrigin(mean, instr->outputs[1]);
  context.MapOutToOrigin(variance, instr->outputs[2]);
  context.MapOutToOrigin(new_moving_mean, instr->outputs[3]);
  context.MapOutToOrigin(new_moving_variance, instr->outputs[4]);
}

void batch_norm_grad(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 5UL) << " The number of the given inputs is not equal to the required "
                                      << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 3UL) << " The number of the given outputs is not equal to the required"
                                       << instr->op_type;

  auto& y_grad        = instr->inputs[0];
  auto& x             = instr->inputs[1];
  auto& scale         = instr->inputs[2];
  auto& save_mean     = instr->inputs[3];
  auto& save_variance = instr->inputs[4];

  auto epsilon = instr.GetAttrs<float>("epsilon");
  auto layout  = instr.GetAttrs<std::string>("data_layout");

  CinnBuilder* builder = context.builder();
  BatchNormHelper helper(builder, x->shape, scale->shape, layout);

  // bias_grad = reduce_sum(y_grad), shape = [c]
  auto bias_grad = builder->Reduce(y_grad, ReduceKind::kSum, helper.reduce_dim);

  // scale_grad = reduce_sum(y_grad * (x - mean)) * rsqrt(variance + epsilon), shape = [c]
  auto epsilon_1d       = helper.GetTensorFromScalar<float>(epsilon, "epsilon", scale->shape);
  auto variance_add_eps = builder->Add(save_variance, epsilon_1d);
  auto std_variance_inv = builder->Rsqrt(variance_add_eps);

  auto mean_4d     = builder->BroadcastTo(save_mean, x->shape, {helper.channel_dim});
  auto x_mean_diff = builder->Sub(x, mean_4d);
  auto sum_of_y_grad_mul_x_mean_diff =
      builder->Reduce(builder->Mul(y_grad, x_mean_diff), ReduceKind::kSum, helper.reduce_dim);
  auto scale_grad = builder->Mul(sum_of_y_grad_mul_x_mean_diff, std_variance_inv);

  // x_grad = 1/nhw * scale * rsqrt(variance + epsilon) *
  //   (nhw * y_grad - reduce_sum(y_grad) - (x - mean) * reduce_sum(y_grad * (x - mean)) / (variance + epsilon))
  // => x_grad = tmp0 * (tmp1 - tmp2 - tmp3)
  auto element_count_1d = helper.GetTensorFromScalar<float>(helper.element_count, "element_count_1d", scale->shape);
  auto scaled_std_variance_inv = builder->Mul(scale, std_variance_inv);
  auto tmp0 =
      builder->BroadcastTo(builder->Div(scaled_std_variance_inv, element_count_1d), x->shape, {helper.channel_dim});

  auto element_count_4d = helper.GetTensorFromScalar<float>(helper.element_count, "element_count_4d", x->shape);
  auto tmp1             = builder->Mul(y_grad, element_count_4d);

  auto tmp2 = builder->BroadcastTo(bias_grad, x->shape, {helper.channel_dim});

  auto sum_of_y_grad_mul_x_mean_diff_4d =
      builder->BroadcastTo(sum_of_y_grad_mul_x_mean_diff, x->shape, {helper.channel_dim});
  auto tmp3_0              = builder->Mul(x_mean_diff, sum_of_y_grad_mul_x_mean_diff_4d);
  auto variance_add_eps_4d = builder->BroadcastTo(variance_add_eps, x->shape, {helper.channel_dim});
  auto tmp3                = builder->Div(tmp3_0, variance_add_eps_4d);

  auto x_grad = builder->Mul(tmp0, builder->Sub(builder->Sub(tmp1, tmp2), tmp3));

  context.MapOutToOrigin(x_grad, instr->outputs[0]);
  context.MapOutToOrigin(scale_grad, instr->outputs[1]);
  context.MapOutToOrigin(bias_grad, instr->outputs[2]);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(batch_norm_train_decomposer) {
  CINN_DECOMPOSER_REGISTER(batch_norm_train, cinn::frontend::decomposer::batch_norm_train);

  return true;
}

CINN_REGISTER_HELPER(batch_norm_grad_decomposer) {
  CINN_DECOMPOSER_REGISTER(batch_norm_grad, cinn::frontend::decomposer::batch_norm_grad);

  return true;
}

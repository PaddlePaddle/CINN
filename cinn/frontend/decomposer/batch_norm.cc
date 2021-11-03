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

  CHECK_EQ(x->shape.size(), 4UL) << "Only 4-D input tensor is supported, but get " << x->shape.size()
                                 << "-D input tensor.";
  CinnBuilder* builder        = context.builder();
  std::vector<int> reduce_dim = {};
  float element_count         = 0;
  int channel_dim             = 0;
  if (layout == "NCHW") {
    channel_dim   = 1;
    reduce_dim    = {0, 2, 3};
    element_count = x->shape[0] * x->shape[2] * x->shape[3];
  } else if (layout == "NHWC") {
    channel_dim   = 3;
    reduce_dim    = {0, 1, 2};
    element_count = x->shape[0] * x->shape[1] * x->shape[2];
  } else {
    LOG(FATAL) << layout << " setting is not support!";
  }

  // shape [c]
  auto element_count_broadcast = builder->BroadcastTo(
      builder->ConstScalar<float>(element_count, common::UniqName("element_count")), scale->shape, {0});
  auto epsilon_broadcast =
      builder->BroadcastTo(builder->ConstScalar<float>(epsilon, common::UniqName("epsilon")), scale->shape, {0});

  // batch norm train
  // mean            = reduce_mean(x)
  // mean_square     = reduce_mean(x * x)
  // variance        = mean_square - mean * mean
  // std_variance    = sqrtf(variance + epsilon)
  // y               = scale * (x - mean) / std_variance + bias
  // moving_mean     = moving_mean * momentum + (1.0 - momentum) * mean
  // moving_variance = moving_variance * momentum + (1.0 - momentum) * variance

  // compute saved mean, shape = [c]
  auto sum     = builder->Reduce(x, ReduceKind::kSum, reduce_dim);
  auto mean    = builder->Div(sum, element_count_broadcast);
  auto mean_4d = builder->BroadcastTo(mean, x->shape, {channel_dim});

  // compute saved variance, shape = [c], simplified by equation: E(x^2) - [E(x)]^2
  auto x_square      = builder->Mul(x, builder->Identity(x));
  auto x_square_sum  = builder->Reduce(x_square, ReduceKind::kSum, reduce_dim);
  auto x_square_mean = builder->Div(x_square_sum, element_count_broadcast);

  auto variance        = builder->Sub(x_square_mean, builder->Mul(mean, builder->Identity(mean)));
  auto std_variance    = builder->Sqrt(builder->Add(variance, epsilon_broadcast));
  auto std_variance_4d = builder->BroadcastTo(std_variance, x->shape, {channel_dim});

  // y = scale * (x - mean) / std_variance + bias
  auto scale_4d    = builder->BroadcastTo(scale, x->shape, {channel_dim});
  auto bias_4d     = builder->BroadcastTo(bias, x->shape, {channel_dim});
  auto scaled_diff = builder->Mul(scale_4d, builder->Sub(x, mean_4d));
  auto y_nobias    = builder->Div(scaled_diff, std_variance_4d);
  auto y           = builder->Add(y_nobias, bias_4d);

  // shape = [c]
  auto factor_0 = builder->BroadcastTo(
      builder->ConstScalar<float>(momentum, common::UniqName("factor_0")), moving_mean->shape, {0});
  auto factor_1 = builder->BroadcastTo(
      builder->ConstScalar<float>(1.0f - momentum, common::UniqName("factor_1")), moving_variance->shape, {0});

  auto new_moving_mean     = builder->Add(builder->Mul(moving_mean, factor_0), builder->Mul(mean, factor_1));
  auto new_moving_variance = builder->Add(builder->Mul(moving_variance, factor_0), builder->Mul(variance, factor_1));

  // map output id
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

  CHECK_EQ(x->shape.size(), 4UL) << "Only 4-D input tensor is supported, but get " << x->shape.size()
                                 << "-D input tensor.";
  auto epsilon         = instr.GetAttrs<float>("epsilon");
  auto layout          = instr.GetAttrs<std::string>("data_layout");
  CinnBuilder* builder = context.builder();

  std::vector<int> reduce_dim = {};
  float element_count         = 0;
  int channel_dim             = 0;
  if (layout == "NCHW") {
    channel_dim   = 1;
    reduce_dim    = {0, 2, 3};
    element_count = x->shape[0] * x->shape[2] * x->shape[3];
  } else if (layout == "NHWC") {
    channel_dim   = 3;
    reduce_dim    = {0, 1, 2};
    element_count = x->shape[0] * x->shape[1] * x->shape[2];
  } else {
    LOG(FATAL) << layout << " setting is not support!";
  }

  // batch norm grad
  // std_norm = (x - saved_mean) / std_variance
  // y = scale * std_norm + bias
  // ==>
  // bias_grad = reduce_sum(y_grad)
  // scale_grad = reduce_sum(y_grad * std_norm)
  // std_norm_grad = y_grad * scale
  //
  // x_mean_diff = x - saved_mean
  // std_norm = x_mean_diff / std_variance
  // ==>
  // x_mean_diff_grad = std_norm_grad / std_variance
  // std_variance_grad = - std_norm_grad * x_mean_diff / variance
  //
  // variance_grad = 0.5 * std_variance_grad / std_variance
  // mean_grad = variance_grad * -2 * mean - reduce(x_mean_diff_grad)
  // mean_square_grad = variance_grad
  // x_mean_diff_grad += mean_grad
  // x_grad = x_mean_diff_grad + 2 * x * mean_square_grad + mean_grad

  // bias_grad = reduce_sum(dy), shape = [c]
  auto bias_grad = builder->Reduce(y_grad, ReduceKind::kSum, reduce_dim);

  // std_norm = (x - saved_mean) / std_variance
  // scale_grad = y_grad * std_norm, shape = [c]
  auto epsilon_1d = builder->BroadcastTo(builder->ConstScalar(epsilon, common::UniqName("epsilon")), scale->shape, {0});
  auto variance_1d     = builder->Add(save_variance, epsilon_1d);
  auto variance_4d     = builder->BroadcastTo(variance_1d, x->shape, {channel_dim});
  auto std_variance_1d = builder->Sqrt(variance_1d);
  auto std_variance_4d = builder->BroadcastTo(std_variance_1d, x->shape, {channel_dim});

  auto mean_4d     = builder->BroadcastTo(save_mean, x->shape, {channel_dim});
  auto x_mean_diff = builder->Sub(x, mean_4d);
  auto scale_grad =
      builder->Div(builder->Reduce(builder->Mul(y_grad, x_mean_diff), ReduceKind::kSum, reduce_dim), std_variance_1d);

  // std_norm_grad = y_grad * scale, shape = [n,c,h,w]
  auto scale_4d      = builder->BroadcastTo(scale, x->shape, {channel_dim});
  auto std_norm_grad = builder->Mul(y_grad, scale_4d);

  // x_mean_diff_grad = std_norm_grad / std_variance, shape = [n,c,h,w]
  auto x_mean_diff_grad = builder->Div(std_norm_grad, std_variance_4d);  // a portion of x_grad

  // std_variance_grad_1d = - reduce_sum(std_norm_grad * x_mean_diff / variance), shape = [c])
  auto std_variance_grad_1d = builder->Negative(builder->Reduce(
      builder->Div(builder->Mul(std_norm_grad, x_mean_diff), variance_4d), ReduceKind::kSum, reduce_dim));

  // variance = std_variance * std_variance
  // variance_grad = 1/2 * std_variance_grad / std_variance
  auto variance_grad_1d_without_mul = builder->Div(std_variance_grad_1d, std_variance_1d);

  // x_grad_0 = (variance_grad_1d_without_mul * 0.5 / element_count) * 2 * x
  auto element_count_1d =
      builder->BroadcastTo(builder->ConstScalar(element_count, common::UniqName("element_count")), scale->shape, {0});
  auto x_grad_0 = builder->Mul(
      x, builder->BroadcastTo(builder->Div(variance_grad_1d_without_mul, element_count_1d), x->shape, {channel_dim}));

  // -1.0 * mean_grad = ((-1.0 * reduce(x_mean_diff_grad)) + (-1.0 * variance_grad_1d_without_mul * 0.5 * 2 * mean)) /
  // element_count_1d
  auto minus_mean_grad    = builder->Div(builder->Add(builder->Reduce(x_mean_diff_grad, ReduceKind::kSum, reduce_dim),
                                                   builder->Mul(variance_grad_1d_without_mul, save_mean)),
                                      element_count_1d);
  auto minus_mean_grad_4d = builder->BroadcastTo(minus_mean_grad, x->shape, {channel_dim});

  // x_grad = x_mean_diff_grad + mean_grad + x_grad_0
  auto x_grad = builder->Sub(builder->Add(x_mean_diff_grad, x_grad_0), minus_mean_grad_4d);

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

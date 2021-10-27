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

  /*****************batch norm train********************
   * mean = reduce_mean(x)
   * diff = x - mean
   * mean_square = reduce_mean(x*x)
   * variance = mean_square - mean*mean
   * std_var = sqrtf(var)
   * y = diff/std_var * scale + bias
   * moving_mean = moving_mean * factor + (1.0 - factor) * mean
   * moving_variance = moving_variance * factor + (1.0 - factor) * variance
   */
  // compute x sum, shape = [c]
  auto sum     = builder->Reduce(x, ReduceKind::kSum, reduce_dim);
  auto mean    = builder->Div(sum, element_count_broadcast);
  auto mean_4d = builder->BroadcastTo(mean, x->shape, {channel_dim});

  // compute x^2 sum
  auto x_copy        = builder->Identity(x);
  auto x_square      = builder->Mul(x, x_copy);
  auto x_square_sum  = builder->Reduce(x_square, ReduceKind::kSum, reduce_dim);
  auto x_square_mean = builder->Div(x_square_sum, element_count_broadcast);

  // E(x^2) - [E(x)]^2
  auto mean_copy       = builder->Identity(mean);
  auto variance        = builder->Sub(x_square_mean, builder->Mul(mean, mean_copy));
  auto std_variance    = builder->Sqrt(builder->Add(variance, epsilon_broadcast));
  auto std_variance_4d = builder->BroadcastTo(std_variance, x->shape, {channel_dim});
  // y = (x - mean)/var

  auto scale_4d = builder->BroadcastTo(scale, x->shape, {channel_dim});
  auto bias_4d  = builder->BroadcastTo(bias, x->shape, {channel_dim});
  // (x - mean)/var * scale + bias
  auto diff = builder->Sub(x, mean_4d);
  auto y    = builder->Add(bias_4d, builder->Mul(scale_4d, builder->Div(diff, std_variance_4d)));

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

  auto& dy            = instr->inputs[0];
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

  /*****************batch norm grad*********************
   * grad_bias = reduce_sum(dy)
   * grad_scale = reduce_sum(dy * (diff/std_var))
   * grad_std_norm = dy * scale
   * grad_diff = grad_std_norm / std_var
   * grad_std_var = -grad_std_norm * diff / var
   * grad_var = 0.5 * grad_std_var / std_var
   * grad_mean = grad_var * -2 * mean - reduce(grad_diff)
   * grad_mean_square = grad_var
   * grad_diff += grad_mean
   * grad_x = grad_diff + 2 * x * grad_mean_square + grad_mean
   */

  auto epsilon_1d = builder->BroadcastTo(builder->ConstScalar(epsilon, common::UniqName("epsilon")), scale->shape, {0});
  auto element_count_1d = builder->BroadcastTo(
      builder->ConstScalar(1.0f / element_count, common::UniqName("element_count")), scale->shape, {0});
  // grad bias = reduce(dy), shape = [c]
  auto grad_bias = builder->Reduce(dy, ReduceKind::kSum, reduce_dim);

  // grad scale = dy * (x - mean)/var, shape = [c]
  auto mean_4d     = builder->BroadcastTo(save_mean, x->shape, {channel_dim});
  auto variance_1d = builder->Add(save_variance, epsilon_1d);
  auto variance_4d = builder->BroadcastTo(variance_1d, x->shape, {channel_dim});
  // std variance
  auto std_variance_1d = builder->Sqrt(variance_1d);
  auto std_variance_4d = builder->BroadcastTo(std_variance_1d, x->shape, {channel_dim});

  auto diff = builder->Sub(x, mean_4d);
  // grad scale = dy * (diff/std_var), shape = [c]
  auto grad_scale =
      builder->Reduce(builder->Mul(dy, builder->Div(diff, std_variance_4d)), ReduceKind::kSum, reduce_dim);

  // grad [(x - mean)/std_var] = dy * scale, shape = [n,c,h,w]
  auto scale_4d      = builder->BroadcastTo(scale, x->shape, {channel_dim});
  auto grad_std_norm = builder->Mul(dy, scale_4d);

  // grad [diff=(x - mean)] = dstd/std_var, shape = [n,c,h,w]
  auto grad_diff = builder->Div(grad_std_norm, std_variance_4d);

  // grad std var = -1 * reduce((grad_std * diff) / (var), shape = [c])
  auto grad_std_variance_1d = builder->Negative(
      builder->Reduce(builder->Div(builder->Mul(grad_std_norm, diff), variance_4d), ReduceKind::kSum, reduce_dim));

  // grad var = 1/2 * dy / std_var, do not mul 0.5 first
  auto grad_variance_1d_without_mul = builder->Div(grad_std_variance_1d, std_variance_1d);

  // grad_x0 = broadcastTo(grad_variance_1d_without_mul * 0.5 /element_count) * 2 * x
  auto grad_x0 = builder->Mul(
      x, builder->BroadcastTo(builder->Mul(grad_variance_1d_without_mul, element_count_1d), x->shape, {channel_dim}));

  // -1.0 * grad_mean = ( -1.0 * reduce(grad_diff) + -1.0 * grad_variance_1d_without_mul * 0.5 * 2 * mean) /
  // element_count_1d
  auto minus_grad_mean = builder->Mul(element_count_1d,
                                      builder->Add(builder->Reduce(grad_diff, ReduceKind::kSum, reduce_dim),
                                                   builder->Mul(grad_variance_1d_without_mul, save_mean)));

  // grad_x = grad_diff + boradcastTo(grad_mean) + grad_x0
  auto grad_x =
      builder->Sub(builder->Add(grad_diff, grad_x0), builder->BroadcastTo(minus_grad_mean, x->shape, {channel_dim}));

  // set output
  context.MapOutToOrigin(grad_x, instr->outputs[0]);
  context.MapOutToOrigin(grad_scale, instr->outputs[1]);
  context.MapOutToOrigin(grad_bias, instr->outputs[2]);
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

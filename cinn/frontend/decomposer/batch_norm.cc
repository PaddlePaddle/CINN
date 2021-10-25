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
  CHECK_EQ(instr->inputs.size(), 5UL) << " 5 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 5UL) << "5 output tensor for " << instr->op_type;

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
   * diff2 = diff * diff
   * var = reduce_mean(diff2)
   * std_var = sqrtf(var)
   * y = diff/std_var * scale + bias
   * running_mean = running_mean * factor + (1.0 - factor) * mean
   * running_var = running_var * factor + (1.0 - factor) * var
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
  y.set_id(common::UniqName("batch_norm_train_output"));

  // shape = [c]
  auto factor_0 = builder->BroadcastTo(
      builder->ConstScalar<float>(momentum, common::UniqName("factor_0")), moving_mean->shape, {0});
  auto factor_1 = builder->BroadcastTo(
      builder->ConstScalar<float>(1.0f - momentum, common::UniqName("factor_1")), moving_variance->shape, {0});

  auto new_moving_mean = builder->Add(builder->Mul(moving_mean, factor_0), builder->Mul(mean, factor_1));
  new_moving_mean.set_id(common::UniqName("new_moving_mean"));
  auto new_moving_variance = builder->Add(builder->Mul(moving_variance, factor_0), builder->Mul(variance, factor_1));
  new_moving_variance.set_id(common::UniqName("new_moving_variance"));

  // map output id
  context.MapOutToOrigin(y, instr->outputs[0]);
  context.MapOutToOrigin(mean, instr->outputs[1]);
  context.MapOutToOrigin(variance, instr->outputs[2]);
  context.MapOutToOrigin(new_moving_mean, instr->outputs[3]);
  context.MapOutToOrigin(new_moving_variance, instr->outputs[4]);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(batch_norm_train_decomposer) {
  CINN_DECOMPOSER_REGISTER(batch_norm_train, cinn::frontend::decomposer::batch_norm_train);

  return true;
}

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

namespace cinn {
namespace frontend {
namespace decomposer {

void batch_norm_train(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 5UL) << " 5 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 5UL) << "5 output tensor for " << instr->op_type;

  auto& x            = instr->inputs[0];
  auto& scale        = instr->inputs[1];
  auto& bias         = instr->inputs[2];
  auto& running_mean = instr->inputs[3];
  auto& running_var  = instr->inputs[4];

  float epsilon        = instr.GetAttrs<float>("epsilon");
  std::string layout   = instr.GetAttrs<std::string>("layout");
  float running_factor = instr.GetAttrs<float>("running_factor");

  CinnBuilder* builder   = context.builder_;
  std::vector<int> r_dim = {};
  float element_count    = 0;
  int c_dim              = 0;
  if (layout == "NCHW") {
    c_dim         = 1;
    r_dim         = {0, 2, 3};
    element_count = x->shape[0] * x->shape[2] * x->shape[3];
  } else if (layout == "NHWC") {
    c_dim         = 3;
    r_dim         = {0, 1, 2};
    element_count = x->shape[0] * x->shape[1] * x->shape[2];
  } else {
    LOG(FATAL) << layout << " setting is not support!";
  }

  // shape [c]
  auto v_element_count = builder->BroadcastTo(
      builder->ConstScalar<float>(element_count, common::UniqName("element_count")), scale->shape, {0});
  auto v_epsilon =
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

  // compute sum, shape = [c]
  auto sum = builder->Reduce(x, ReduceKind::kSum, r_dim);
  // compute mean = [c] -> [n, c, h, w]
  auto save_mean = builder->Div(sum, v_element_count);
  auto mean      = builder->BroadcastTo(save_mean, x->shape, {c_dim});
  // diff
  auto diff = builder->Sub(x, mean);
  auto _diff = builder->Identity(diff);
  // diff2
  auto diff2 = builder->Mul(diff, _diff);

  // sum variance, shape = [c]
  auto sum_diff2 = builder->Reduce(diff2, ReduceKind::kSum, r_dim);
  // variance, shape[c]
  auto var2 = builder->Div(sum_diff2, v_element_count);
  // standard variance, shape[c] -> [n, c, h, w]
  auto save_var = builder->Sqrt(var2);
  auto var      = builder->BroadcastTo(builder->Add(save_var, v_epsilon), x->shape, {c_dim});

  auto v_scale = builder->BroadcastTo(scale, x->shape, {c_dim});
  auto v_bias  = builder->BroadcastTo(bias, x->shape, {c_dim});
  // (x - mean)/var * scale + bias
  auto y = builder->Add(v_bias, builder->Mul(v_scale, builder->Div(diff, var)));

  // shape = [c]
  auto factor_0 = builder->BroadcastTo(
      builder->ConstScalar<float>(running_factor, common::UniqName("factor_0")), running_mean->shape, {0});
  auto factor_1 = builder->BroadcastTo(
      builder->ConstScalar<float>(1.0f - running_factor, common::UniqName("factor_1")), running_var->shape, {0});
  auto new_mean = builder->Add(builder->Mul(running_mean, factor_0), builder->Mul(save_mean, factor_1));
  auto new_var  = builder->Add(builder->Mul(running_var, factor_0), builder->Mul(save_var, factor_1));

  // map output id
  context.MapVarToOrigin(y, instr->outputs[0]);
  context.MapVarToOrigin(save_mean, instr->outputs[1]);
  context.MapVarToOrigin(save_var, instr->outputs[2]);
  context.MapVarToOrigin(new_mean, instr->outputs[3]);
  context.MapVarToOrigin(new_var, instr->outputs[4]);
}

void batch_norm_grad(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 5UL) << " 5 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 3UL) << "3 output tensor for " << instr->op_type;

  auto& x         = instr->inputs[0];
  auto& dy        = instr->inputs[1];
  auto& scale     = instr->inputs[2];
  auto& save_mean = instr->inputs[3];
  auto& save_var  = instr->inputs[4];

  auto layout = instr.GetAttrs<std::string>("layout");

  CinnBuilder* builder = context.builder_;

  std::vector<int> r_dim = {};
  float element_count    = 0;
  int c_dim              = 0;
  if (layout == "NCHW") {
    c_dim         = 1;
    r_dim         = {0, 2, 3};
    element_count = x->shape[0] * x->shape[2] * x->shape[3];
  } else if (layout == "NHWC") {
    c_dim         = 3;
    r_dim         = {0, 1, 2};
    element_count = x->shape[0] * x->shape[1] * x->shape[2];
  } else {
    LOG(FATAL) << layout << " setting is not support!";
  }

  /*****************batch norm train********************
  * mean = reduce_mean(x)
  * diff = x - mean
  * diff2 = diff * diff
  * var = reduce_mean(diff2)
  * std_var = sqrtf(var)
  * y = diff/std_var * scale + bias
  */

  // grad bias = reduce(dy), shape = [c]
  auto grad_bias = builder->Reduce(dy, ReduceKind::kSum, r_dim);
  // grad scale = dy * (x - mean)/var, shape = [c]
  auto mean = builder->BroadcastTo(save_mean, x->shape, {c_dim});
  auto var  = builder->BroadcastTo(save_var, x->shape, {c_dim});

  auto diff = builder->Sub(x, mean);
  // grad scale = dy * (diff/var), shape = [c]
  auto grad_scale = builder->Reduce(builder->Mul(dy, builder->Div(diff, var)), ReduceKind::kSum, r_dim);
  // grad [(x - mean)/var] = dy * scale, shape = [n,c,h,w]
  auto v_scale  = builder->BroadcastTo(scale, x->shape, {c_dim});
  auto grad_std = builder->Mul(dy, v_scale);

  // grad [diff=(x - mean)] = dstd/var, shape = [n,c,h,w]
  auto grad_diff0 = builder->Div(grad_std, var);
  auto _var = builder->Identity(var); 
  // grad var = Negative((grad_std * diff) / (save_var*save_var)), shape = [c]
  auto grad_var = builder->Negative(
      builder->Reduce(builder->Div(builder->Mul(grad_std, diff), builder->Mul(var, _var)), ReduceKind::kSum, r_dim));
  // grad diff2 = (1.0f / ( 2 * num_element)) * (grad_var / save_var), shape[n,c,h,w]
  auto v_element_count = builder->BroadcastTo(
      builder->ConstScalar(1.0f / element_count, common::UniqName("element_count")), grad_var->shape, {0});
  auto grad_diff2 =
      builder->BroadcastTo(builder->Mul(v_element_count, builder->Div(grad_var, save_var)), x->shape, {c_dim});
  // grad diff = (grad_diff2 * 2 * diff), shape = [n,c,h,w]
  auto grad_diff = builder->Add(builder->Mul(grad_diff2, diff), grad_diff0);
  // grad mean, shape = [c]
  auto v_minus_element_count = builder->BroadcastTo(
      builder->ConstScalar(-1.0f / element_count, common::UniqName("minus_element_count")), grad_diff->shape, {0});
  // grad_sum = -1 * grad_diff / element_count
  auto grad_sum = builder->Reduce(builder->Mul(v_minus_element_count, grad_diff), ReduceKind::kSum, r_dim);
  // grad x
  auto grad_x = builder->Add(grad_diff, builder->BroadcastTo(grad_sum, x->shape, {c_dim}));

  // set output
  context.MapVarToOrigin(grad_x, instr->outputs[0]);
  context.MapVarToOrigin(grad_scale, instr->outputs[1]);
  context.MapVarToOrigin(grad_bias, instr->outputs[2]);
}

void conv2d_grad(const Instruction& instr, const DecomposerContext& context) {
  auto& x  = instr->inputs[0];
  auto& w  = instr->inputs[1];
  auto& dy = instr->inputs[2];

  CinnBuilder* builder = context.builder_;
  // create backward data
  auto dx = builder->Conv(w,
                          dy,
                          instr.GetAttrs<std::vector<int>>("stride"),
                          instr.GetAttrs<std::vector<int>>("padding"),
                          instr.GetAttrs<std::vector<int>>("dilation"),
                          instr.GetAttrs<int>("groups"),
                          "backward_data",
                          instr.GetAttrs<std::string>("layout"),
                          instr.GetAttrs<std::string>("padding_algorithm"));
  context.MapVarToOrigin(dx, instr->outputs[0]);

  // create backward filter
  auto dw = builder->Conv(x,
                          dy,
                          instr.GetAttrs<std::vector<int>>("stride"),
                          instr.GetAttrs<std::vector<int>>("padding"),
                          instr.GetAttrs<std::vector<int>>("dilation"),
                          instr.GetAttrs<int>("groups"),
                          "backward_filter",
                          instr.GetAttrs<std::string>("layout"),
                          instr.GetAttrs<std::string>("padding_algorithm"),
                          w->shape);
  context.MapVarToOrigin(dw, instr->outputs[1]);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(nn) {
  CINN_DECOMPOSER_REGISTER(batch_norm_train, cinn::frontend::decomposer::batch_norm_train);
  CINN_DECOMPOSER_REGISTER(batch_norm_grad, cinn::frontend::decomposer::batch_norm_grad);
  CINN_DECOMPOSER_REGISTER(conv2d_grad, cinn::frontend::decomposer::conv2d_grad);

  return true;
}

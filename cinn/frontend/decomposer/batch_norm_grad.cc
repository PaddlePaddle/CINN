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

void batch_norm_grad(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 5UL) << " 5 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 3UL) << "3 output tensor for " << instr->op_type;

  auto& x         = instr->inputs[0];
  auto& dy        = instr->inputs[1];
  auto& scale     = instr->inputs[2];
  auto& save_mean = instr->inputs[3];
  auto& save_var  = instr->inputs[4];

  CHECK_EQ(x->shape.size(), 4UL) << "Only 4-D input tensor is supported, but get " << x->shape.size()
                                 << "-D input tensor.";
  auto layout          = instr.GetAttrs<std::string>("layout");
  CinnBuilder* builder = context.builder();

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
  auto _var       = builder->Identity(var);
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
      builder->ConstScalar(-1.0f / element_count, common::UniqName("minus_element_count")), scale->shape, {0});
  // grad_sum = -1 * grad_diff / element_count
  auto grad_sum = builder->Mul(v_minus_element_count, builder->Reduce(grad_diff, ReduceKind::kSum, r_dim));
  // grad x
  auto grad_x = builder->Add(grad_diff, builder->BroadcastTo(grad_sum, x->shape, {c_dim}));

  // set output
  context.MapOutToOrigin(grad_x, instr->outputs[0]);
  context.MapOutToOrigin(grad_scale, instr->outputs[1]);
  context.MapOutToOrigin(grad_bias, instr->outputs[2]);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(batch_norm_grad_decomposer) {
  CINN_DECOMPOSER_REGISTER(batch_norm_grad, cinn::frontend::decomposer::batch_norm_grad);

  return true;
}

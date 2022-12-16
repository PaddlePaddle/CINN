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

void softmax(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 1UL) << "1 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL) << "1 output tensor for " << instr->op_type;

  auto* builder = context.builder();
  /*
  // softmax = e(x) / sum(e(x))
  // as e(x) may cause overflow, using (e(x) / e(max(x))) / (sum(e(x) /  e(max(x))).
  // do simplify : e(x - max(x)) / sum(e(x - max(x)))

  // max_x = reduce_max(x)
  // x_sub_max = x - max_x
  // exp_x = e^x
  // sum_exp_x = reduce_sum(exp_x)
  // out = exp_x / sum_exp_x
  */
  int axis = instr.GetAttrs<float>("axis");
  auto x   = instr->inputs[0];

  auto max_x     = builder->ReduceMax(x, {axis}, true);
  auto x_sub_max = builder->Subtract(x, max_x_b, axis);
  auto exp_x     = builder->Exp(x_sub_max);
  auto sum_exp_x = builder->ReduceSum(exp_x, {axis}, true);
  auto out       = builder->Divide(exp_x, sum_exp_x, axis);

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(out, instr->outputs[0]);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

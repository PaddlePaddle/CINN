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

void relu(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 1UL) << " 1 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL) << "1 output tensor for " << instr->op_type;
  auto x        = instr->inputs[0];
  auto output   = instr->outputs[0];
  auto* builder = context.builder();

  auto bcast_zero = builder->FillConstant<float>(x->shape, 0.0f, common::UniqName("zero"));
  auto out        = builder->Max(x, bcast_zero);

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(out, output);
}

void relu_grad(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 2UL) << " 2 input tensors for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL) << "1 output tensor for " << instr->op_type;
  auto dout     = instr->inputs[0];
  auto out      = instr->inputs[1];
  auto dx       = instr->outputs[0];
  auto* builder = context.builder();

  auto bcast_zero = builder->FillConstant<float>(out->shape, 0.0f, common::UniqName("zero"));
  auto condition  = builder->GreaterThan(out, bcast_zero);
  auto res        = builder->Select(condition, dout, bcast_zero);

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(res, dx);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(activation_decomposers) {
  CINN_DECOMPOSER_REGISTER(relu, cinn::frontend::decomposer::relu);

  return true;
}

CINN_REGISTER_HELPER(activation_grad_decomposers) {
  CINN_DECOMPOSER_REGISTER(relu_grad, cinn::frontend::decomposer::relu_grad);

  return true;
}

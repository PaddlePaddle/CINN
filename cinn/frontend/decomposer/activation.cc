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
  auto x       = instr->inputs[0];
  auto output  = instr->outputs[0];
  auto builder = context.builder_;

  auto zero_var   = builder->ConstScalar<float>(0.f, common::UniqName("zero"));
  auto bcast_zero = builder->BroadcastTo(zero_var, x->shape, {0});
  auto out        = builder->Max(x, bcast_zero);

  // set the original output to the output of decomposed operator.
  auto max_instr        = builder->GetInstruction(builder->NumInstructions() - 1);
  max_instr->outputs[0] = output;
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(activation) {
  CINN_DECOMPOSER_REGISTER(relu, ::cinn::common::DefaultHostTarget()).SetBody(cinn::frontend::decomposer::relu);
  CINN_DECOMPOSER_REGISTER(relu, ::cinn::common::DefaultNVGPUTarget()).SetBody(cinn::frontend::decomposer::relu);

  return true;
}

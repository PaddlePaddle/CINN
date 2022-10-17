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

void sum(const Instruction& instr, const DecomposerContext& context) {
  CHECK_GT(instr->inputs.size(), 0UL) << "At least 1 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL) << "1 output tensor for " << instr->op_type;
  auto inputs   = instr->inputs;
  auto output   = instr->outputs[0];
  auto* builder = context.builder();

  auto sum = builder->Identity(inputs[0]);
  for (size_t i = 1; i < inputs.size(); ++i) {
    sum = builder->Add(sum, inputs[i]);
  }

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(sum, output);
}

void clip(const Instruction& instr, const DecomposerContext& context) {
  CHECK_GT(instr->inputs.size(), 0UL) << "At least 1 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL) << "1 output tensor for " << instr->op_type;

  auto input    = instr->inputs[0];
  auto output   = instr->outputs[0];
  auto* builder = context.builder();

  auto max_val  = instr.GetAttrs<float>("max_val");
  auto min_val  = instr.GetAttrs<float>("min_val");
  auto max_val_ = builder->FillConstant(input->shape, max_val, common::UniqName("constant"));
  auto min_val_ = builder->FillConstant(input->shape, min_val, common::UniqName("constant"));

  auto out0 = builder->Min(input, max_val_);
  auto out1 = builder->Max(out0, min_val_);
  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(out1, output);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(sum_decomposers) {
  CINN_DECOMPOSER_REGISTER(sum, cinn::frontend::decomposer::sum);

  return true;
}

CINN_REGISTER_HELPER(clip_decomposers) {
  CINN_DECOMPOSER_REGISTER(clip, cinn::frontend::decomposer::clip);

  return true;
}

// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/program_pass.h"

namespace cinn::frontend::pass {

class ReductionRewritePass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

  void ApplyImpl(Program* program, const Target& target) const {
    CinnBuilder builder("reduction_rewrite_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (size_t i = 0; i < program->size(); ++i) {
      const auto& instr = (*program)[i];
      if (CanRewriteReduction(instr)) {
        builder.AppendInstruction(RewriteReduction(instr));
      } else {
        builder.AppendInstruction(instr);
      }
    }
    *program = builder.Build();
  }

 private:
  bool CanRewriteReduction(const Instruction& instr) const { return false; }

  Instruction RewriteReduction(const Instruction& instr) const { return instr; }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(ReductionRewrite) {
  CINN_REGISTER_PROGRAM_PASS(ReductionRewrite, ::cinn::frontend::pass::ReductionRewritePass);

  return true;
}

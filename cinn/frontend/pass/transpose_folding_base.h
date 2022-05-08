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

#include <absl/container/flat_hash_map.h>

#include <string>
#include <unordered_set>

#include "cinn/common/target.h"
#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

class TransposeFoldingBase : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  virtual void set_target_instrs() = 0;

  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    VLOG(4) << "-- Before folding: " << *program;
    set_target_instrs();
    // `out2instr` is used to represent the mapping of Output to Instruction.
    absl::flat_hash_map<std::string, Instruction*> out2instr;
    // `in2instr` is used to represent the mapping of Input to Instruction.
    absl::flat_hash_map<std::string, std::unordered_set<Instruction*>> in2instr;
    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& out : instr->outputs) {
        out2instr[out->id] = &instr;
      }
      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
    }

    // `remove_instrs` is used to represent Instructions of which type is transpose to be deleted.
    absl::flat_hash_set<Instruction*> remove_instrs;
    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      if (target_instrs_.count(instr->op_type)) {
        FoldTranspose(&instr, out2instr, in2instr, fetch_ids, &remove_instrs);
      }
    }

    CinnBuilder builder("transpose_folding_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (!remove_instrs.count(&(*program)[i])) {
        builder.AppendInstruction((*program)[i]);
      }
    }
    *program = builder.Build();
    VLOG(4) << "-- After folding: " << *program;
  }

  virtual void FoldTranspose(Instruction* instr,
                             const absl::flat_hash_map<std::string, Instruction*>& out2instr,
                             const absl::flat_hash_map<std::string, std::unordered_set<Instruction*>>& in2instr,
                             const std::unordered_set<std::string>& fetch_ids,
                             absl::flat_hash_set<Instruction*>* remove_instrs) const = 0;

  bool IsValidTranspose(const Instruction& transpose) const {
    if ("transpose" != transpose->op_type) {
      return false;
    }

    // `axis` of tranpose must be consecutive in the reverse order,
    // excluding the first dim
    auto axis = transpose.GetAttrs<std::vector<int>>("axis");
    if (axis[0] == 0) {
      // In the batched martix multiplication, the first dim should be batch dim.
      for (size_t i = 1; i < axis.size(); ++i) {
        if (axis[i] != axis.size() - i) {
          return false;
        }
      }
      return true;
    } else if (axis[0] == axis.size() - 1) {
      // Otherwise, the axis should be consecutive in the reverse order.
      for (size_t i = 1; i < axis.size(); ++i) {
        if (axis[i] != axis.size() - 1 - i) {
          return false;
        }
      }
      return true;
    }

    // if axis[0] not 0 or axis.size() - 1, cannot folding
    return false;
  }

  std::unordered_set<std::string> target_instrs_;
};

}  // namespace cinn::frontend::pass

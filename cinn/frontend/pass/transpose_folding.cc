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

// Pass `TransposeFolding` folds transpose into dot, then both of them can be implemented by a
// GEMM kernel. For each dot operator, try folding every input that belong output of transpose.
// If output of tranpose in `fetch_ids`, keep the operator.
class TransposeFoldingPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) const override {
    // `out2instr` is used to represent the mapping of Output to Instruction.
    absl::flat_hash_map<std::string, Instruction*> out2instr;
    // `in2instr` is used to represent the mapping of Input to Instruction.
    absl::flat_hash_map<std::string, std::unordered_set<Instruction*>> in2instr;
    // `remove_instrs` is used to represent Instructions which type is transpose to be deleted.
    absl::flat_hash_set<Instruction*> remove_instrs;
    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& out : instr->outputs) {
        out2instr[out->id] = &instr;
      }
      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
      // Operator dot is actually operator matmul.
      if ("matmul" == instr->op_type) {
        for (auto transpose : TryFoldIntoDot(&instr, out2instr, in2instr)) {
          if (fetch_ids.find((*transpose)->outputs[0]->id) == fetch_ids.end()) {
            remove_instrs.insert(transpose);
          }
        }
      } else {
        // The output of transpose maybe used by other operators.
        for (const auto& in : instr->inputs) {
          auto iter = out2instr.find(in->id);
          if (iter != out2instr.end()) {
            remove_instrs.erase(iter->second);
          }
        }
      }
    }
    CinnBuilder builder("transpose_folding_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (remove_instrs.end() != remove_instrs.find(&(*program)[i])) continue;
      builder.AppendInstruction((*program)[i]);
    }
    *program = builder.Build();
  }

 private:
  // Rules that transpose can be folded into dot:
  //   1) input operand of dot must be transpose;
  //   2) `axis` of tranpose must be consecutive in the reverse order, excluding the first dim;
  std::vector<Instruction*> TryFoldIntoDot(
      Instruction* dot,
      const absl::flat_hash_map<std::string, Instruction*>& out2instr,
      const absl::flat_hash_map<std::string, std::unordered_set<Instruction*>>& in2instr) const {
    std::vector<Instruction*> remove_instrs;
    if (!(*dot)->attrs.empty()) return remove_instrs;
    auto is_transpose = [](const Instruction& transpose) {
      if ("transpose" != transpose->op_type) {
        return false;
      }

      // The following codes for Rule 2).
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

      // if axis[0] not 0 or -1, cannot folding
      return false;
    };
    for (size_t i = 0; i < (*dot)->inputs.size(); ++i) {
      auto transpose_out_name = (*dot)->inputs[i]->id;
      auto iter               = out2instr.find(transpose_out_name);
      if (iter != out2instr.end()) {
        // the previous op of matmul
        const auto& operand = *(iter->second);
        if (is_transpose(operand)) {
          // x-> transpose -> out -> dot => x -> dot
          (*dot)->inputs[i] = operand->inputs[0];
          (*dot).SetAttr(i == 0 ? "trans_a" : "trans_b", true);

          CHECK(in2instr.find(transpose_out_name) != in2instr.end())
              << "The var [" << transpose_out_name
              << "] should be someone op's input, but couldn't found ! Please check.";
          CHECK(in2instr.at(transpose_out_name).count(dot))
              << "The var [" << transpose_out_name << "] should be matmul's input, but couldn't found ! Please check.";

          if (in2instr.at(transpose_out_name).size() == 1) {
            // the transpose is only link to matmul, should remove
            remove_instrs.push_back(iter->second);
          }
        }
      }
    }
    return remove_instrs;
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(TransposeFolding) {
  CINN_REGISTER_PROGRAM_PASS(TransposeFolding, ::cinn::frontend::pass::TransposeFoldingPass);

  return true;
}

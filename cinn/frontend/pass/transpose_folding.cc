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

// Rules that transpose can fold into dot:
//   1) input operand of dot must be transpose;
//   2) `axis` of tranpose must be consecutive;
std::vector<Instruction*> TryFoldIntoDot(Instruction* dot,
                                         const absl::flat_hash_map<std::string, Instruction*>& var_instrs) {
  std::vector<Instruction*> remove_instrs;
  if (!(*dot)->attrs.empty()) return remove_instrs;
  auto is_transpose = [](const Instruction& transpose) {
    if ("transpose" != transpose->op_type) return false;
    auto axis = transpose.GetAttrs<std::vector<int>>("axis");
    for (size_t i = 0; i < axis.size(); ++i) {
      if (axis[i] + 1 != axis.size() - i) return false;
    }
    return true;
  };
  for (size_t i = 0; i < (*dot)->inputs.size(); ++i) {
    auto iter = var_instrs.find((*dot)->inputs[i]->id);
    if (iter != var_instrs.end()) {
      const auto& operand = *(iter->second);
      if (is_transpose(operand)) {
        // x-> transpose -> out -> dot => x -> dot
        (*dot)->inputs[i] = operand->inputs[0];
        (*dot).SetAttr(i == 0 ? "trans_a" : "trans_b", true);
        remove_instrs.push_back(iter->second);
      }
    }
  }
  return remove_instrs;
}

// Pass `TransposeFolding` folds transpose into dot, than it can be implemented by a GEMM kernel.
// For each dot operator, try folding every input that belong output of transpose. If output of
// tranpose in `fetch_ids`, keep the operator.
void TransposeFolding(Program* program, const std::unordered_set<std::string>& fetch_ids) {
  absl::flat_hash_map<std::string, Instruction*> var_instrs;
  absl::flat_hash_set<Instruction*> remove_instrs;
  for (size_t i = 0; i < program->size(); ++i) {
    auto& instr = (*program)[i];
    for (const auto& out : instr->outputs) {
      var_instrs[out->id] = &instr;
    }
    // Operator dot is actually operator matmul.
    if ("matmul" == instr->op_type) {
      for (auto transpose : TryFoldIntoDot(&instr, var_instrs)) {
        if (fetch_ids.find((*transpose)->outputs[0]->id) == fetch_ids.end()) {
          remove_instrs.insert(transpose);
        }
      }
    } else {
      // The output of transpose maybe used by other operators.
      for (const auto& in : instr->inputs) {
        auto iter = var_instrs.find(in->id);
        if (iter != var_instrs.end()) {
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

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(TransposeFolding) {
  CINN_REGISTER_PROGRAM_PASS_FUNCTION(TransposeFolding)
      .describe("This pass folds transpose into dot.")
      .set_body(cinn::frontend::pass::TransposeFolding);

  return true;
}

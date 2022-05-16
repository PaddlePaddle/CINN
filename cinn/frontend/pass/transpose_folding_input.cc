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
#include <absl/types/variant.h>

#include <string>
#include <unordered_set>

#include "cinn/frontend/pass/transpose_folding_base.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

// Pass `TransposeFoldingInput` folds transpose into dot, then both of them can be implemented by a
// GEMM kernel. For each dot operator, try folding every input that belong output of transpose.
// If output of tranpose in `fetch_ids`, keep the operator.
class TransposeFoldingInputPass : public TransposeFoldingBase {
 public:
  using TransposeFoldingBase::TransposeFoldingBase;

 protected:
  void set_target_instrs() override { TransposeFoldingBase::target_instrs_ = {"matmul"}; }

  // Rules that transpose can be folded into matmul:
  //   1) input operand of dot must be transpose;
  //   2) `axis` of tranpose must be consecutive in the reverse order, excluding the first dim;
  void FoldTranspose(Instruction* dot,
                     const absl::flat_hash_map<std::string, Instruction*>& out2instr,
                     const absl::flat_hash_map<std::string, std::unordered_set<Instruction*>>& in2instr,
                     const std::unordered_set<std::string>& fetch_ids,
                     absl::flat_hash_set<Instruction*>* remove_instrs) const override {
    bool trans_a = false;
    bool trans_b = false;
    if ((*dot)->attrs.contains("trans_a")) {
      trans_a = absl::get<bool>((*dot)->attrs["trans_a"]);
    }
    if ((*dot)->attrs.contains("trans_b")) {
      trans_b = absl::get<bool>((*dot)->attrs["trans_b"]);
    }

    for (size_t i = 0; i < (*dot)->inputs.size(); ++i) {
      auto transpose_out_name = (*dot)->inputs[i]->id;
      auto iter               = out2instr.find(transpose_out_name);
      if (iter != out2instr.end()) {
        // the previous op of matmul
        const auto& operand = *(iter->second);
        if (IsValidTranspose(operand)) {
          // x-> transpose -> out -> dot => x -> dot
          (*dot)->inputs[i] = operand->inputs[0];
          if (i == 0) {
            (*dot).SetAttr("trans_a", static_cast<bool>(trans_a ^ true));
          } else if (i == 1) {
            (*dot).SetAttr("trans_b", static_cast<bool>(trans_b ^ true));
          } else {
            LOG(FATAL) << "The matmul should only have two inputs.";
          }

          CHECK(in2instr.find(transpose_out_name) != in2instr.end())
              << "The var [" << transpose_out_name
              << "] should be someone op's input, but couldn't found ! Please check.";
          const auto& out_instrs = in2instr.at(transpose_out_name);
          CHECK(out_instrs.count(dot)) << "The var [" << transpose_out_name
                                       << "] should be matmul's input, but couldn't found ! Please check.";

          bool can_remove = std::all_of(out_instrs.begin(), out_instrs.end(), [&](Instruction* instr) {
            // the transpose had linked to not matmul op, cannot remove
            return target_instrs_.find((*instr)->op_type) != target_instrs_.end();
          });

          if (can_remove && !fetch_ids.count(transpose_out_name)) {
            // the transpose is only link to matmul and its output is not in fetch_ids, should remove
            remove_instrs->insert(iter->second);
          }
        }
      }
    }
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(TransposeFoldingInput) {
  CINN_REGISTER_PROGRAM_PASS(TransposeFoldingInput, ::cinn::frontend::pass::TransposeFoldingInputPass);

  return true;
}

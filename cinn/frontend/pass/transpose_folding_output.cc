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

class TransposeFoldingOutputPass : public TransposeFoldingBase {
 public:
  using TransposeFoldingBase::TransposeFoldingBase;

 protected:
  void set_target_instrs() { TransposeFoldingBase::target_instrs_ = {"cublas_gemm", "cublas_matmul"}; }

  void FoldTranspose(Instruction* gemm,
                     const absl::flat_hash_map<std::string, Instruction*>& out2instr,
                     const absl::flat_hash_map<std::string, std::unordered_set<Instruction*>>& in2instr,
                     const std::unordered_set<std::string>& fetch_ids,
                     absl::flat_hash_set<Instruction*>* remove_instrs) const override {
    bool trans_out = false;
    if ((*gemm)->attrs.contains("trans_out")) {
      trans_out = absl::get<bool>((*gemm)->attrs["trans_out"]);
    }
    auto gemm_out_name = (*gemm)->outputs[0]->id;
    if (in2instr.contains(gemm_out_name) && in2instr.at(gemm_out_name).size() == 1 && !fetch_ids.count(gemm_out_name)) {
      auto* instr = *(in2instr.at(gemm_out_name).begin());
      if (IsValidTranspose(*instr)) {
        gemm->SetAttr("trans_out", static_cast<bool>(trans_out ^ true));
        (*gemm)->outputs[0] = (*instr)->outputs[0];
        remove_instrs->insert(instr);
      }
    }
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(TransposeFoldingOutput) {
  CINN_REGISTER_PROGRAM_PASS(TransposeFoldingOutput, ::cinn::frontend::pass::TransposeFoldingOutputPass);

  return true;
}

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

#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/program_pass.h"

namespace cinn {
namespace frontend {
namespace pass {

void RemoveIdentity(Program* program, const std::unordered_set<std::string>& fetch_ids) {
  CinnBuilder builder("remove_identity_builder");
  for (auto& var : program->GetInputs()) {
    builder.CreateInput(var);
  }
  absl::flat_hash_set<std::string> inputs;
  absl::flat_hash_set<int> remove_idxs;
  for (int i = program->size() - 1; i >= 0; --i) {
    const auto& instr = (*program)[i];
    for (const auto& in : instr->inputs) {
      inputs.insert(in->id);
    }
    if (instr->op_type != "identity") continue;
    bool can_remove = true;
    for (const auto& out : instr->outputs) {
      if (inputs.end() != inputs.find(out->id) || fetch_ids.end() != fetch_ids.find(out->id)) {
        can_remove = false;
        break;
      }
    }
    if (can_remove) {
      VLOG(2) << "Remove instruction: " << instr;
      remove_idxs.insert(i);
    }
  }
  VLOG(2) << "Total remove " << remove_idxs.size() << " instructions.";
  for (int i = 0; i < program->size(); i++) {
    if (remove_idxs.end() != remove_idxs.find(i)) continue;
    builder.AppendInstruction((*program)[i]);
  }
  *program = builder.Build();
}

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(RemoveIdentity) {
  CINN_REGISTER_PROGRAM_PASS_FUNCTION(RemoveIdentity).set_body(cinn::frontend::pass::RemoveIdentity);

  return true;
}

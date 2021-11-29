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
  CinnBuilder builder("decomposer_builder");
  for (auto& var : program->GetInputs()) {
    builder.CreateInput(var);
  }
  absl::flat_hash_set<size_t> remove_identity_ids;
  absl::flat_hash_map<std::string, size_t> identity_out_id_map;
  for (size_t i = 0; i < program->size(); i++) {
    const auto& instr = (*program)[i];
    if (instr->op_type == "identity") {
      const auto& out = instr->outputs[0];
      if (fetch_ids.end() == fetch_ids.find(out->id)) {
        identity_out_id_map[out->id] = i;
        remove_identity_ids.insert(i);
      }
    } else if (!identity_out_id_map.empty()) {
      const auto& inputs = instr->inputs;
      decltype(identity_out_id_map)::const_iterator iter;
      for (const auto& in : inputs) {
        iter = identity_out_id_map.find(in->id);
        if (identity_out_id_map.end() != iter) break;
      }
      if (identity_out_id_map.end() != iter) {
        const auto& identity_instr = (*program)[iter->second];
        const auto& identity_in    = identity_instr->inputs[0];
        for (const auto& in : inputs) {
          if (identity_in->id == in->id) {
            remove_identity_ids.erase(iter->second);
            identity_out_id_map.erase(iter);
            break;
          }
        }
      }
    }
  }
  for (size_t i = 0; i < program->size(); i++) {
    if (remove_identity_ids.end() != remove_identity_ids.find(i)) continue;
    auto& instr = (*program)[i];
    for (size_t j = 0; j < instr->inputs.size(); j++) {
      auto iter = identity_out_id_map.find(instr->inputs[j]->id);
      if (identity_out_id_map.end() != iter) {
        instr->inputs[j] = (*program)[iter->second]->inputs[0];
      }
    }
    builder.AppendInstruction(instr);
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

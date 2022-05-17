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

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/program_pass.h"
#include "glog/logging.h"

namespace cinn {
namespace frontend {
namespace pass {

// RemoveIdentityPass will remove the identity instructions in following patterns:
//
// 1. When varB is not in fetch_ids, the identity and varB will be removed.
//    When varB is in fetch_ids and varA is not in fetch_ids, the identity and varA will be removed.
//        instrA                      instrA
//          | varA                      |
//      identity           =>           | varA/varB
//          | varB                      |
//        instrB                      instrB
//
// 2. Multiply outputs are also supported.
//        instrA                      instrA
//          | varA                      |
//      identity           =>           | varA/varB
//          | varB                      |
//         / \                         / \
//   instrB   instrC             instrB   instrC
class RemoveIdentityPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    VLOG(5) << "Origin program: " << *program;

    CollectIdentityInstrInfo(*program);
    FindRemoveIdentityInstr(*program, fetch_ids);
    VLOG(3) << "Total remove " << remove_idxs_.size() << " instructions.";
    if (remove_idxs_.size() == 0) {
      return;
    }

    CinnBuilder builder("remove_identity_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); ++i) {
      if (remove_idxs_.count(i)) {
        continue;
      }
      auto& inputs = (*program)[i]->inputs;
      for (size_t j = 0; j < inputs.size(); ++j) {
        if (origin2new_.count(inputs[j].get())) {
          inputs[j] = origin2new_.at(inputs[j].get());
        }
      }
      auto& outputs = (*program)[i]->outputs;
      for (size_t j = 0; j < outputs.size(); ++j) {
        if (origin2new_.count(outputs[j].get())) {
          outputs[j] = origin2new_.at(outputs[j].get());
        }
      }
      builder.AppendInstruction((*program)[i]);
    }
    *program = builder.Build();
    VLOG(5) << "Optimized program: " << *program;

    Clear();
  }

 private:
  void CollectIdentityInstrInfo(const Program& program) {
    std::unordered_map<_Variable_*, Instruction> outputs2instr;
    for (int i = 0; i < program.size(); ++i) {
      auto& instr = program[i];
      for (auto& var : instr->outputs) {
        outputs2instr.emplace(var.get(), instr);
      }
    }

    std::unordered_map<_Instruction_*, int> identity_instr_idx;
    for (int i = 0; i < program.size(); ++i) {
      const auto& instr = program[i];
      if (instr->op_type != "identity") {
        continue;
      }
      CHECK_EQ(instr->inputs.size(), 1) << "identity should have only 1 input.";
      CHECK_EQ(instr->outputs.size(), 1) << "identity should have only 1 output.";

      // Record the index of all identity instructions.
      identity_instr_idx.emplace(instr.get(), i);

      auto& input_var = instr->inputs[0];
      auto iter       = outputs2instr.find(input_var.get());
      bool inserted   = false;
      // Whether input_var is the output of another instruction.
      if (iter != outputs2instr.end()) {
        auto& prev_instr = iter->second;
        if (prev_instr->op_type == "identity") {
          // There are multiple continuous identity instructions, insert the current identity instruction to an existing
          // set.
          for (auto& group : identity_instr_groups_) {
            if (group.count(identity_instr_idx[prev_instr.get()])) {
              group.insert(i);
              inserted = true;
              break;
            }
          }
          CHECK(inserted) << "The previous identity instruction has not be inserted to the identity groups";
        }
      }
      if (!inserted) {
        // Add a new identity group.
        identity_instr_groups_.push_back(std::set<int>{i});
      }
    }
  }

  void FindRemoveIdentityInstr(const Program& program, const std::unordered_set<std::string>& fetch_ids) {
    std::unordered_set<std::string> feed_ids;
    for (auto& var : program.GetInputs()) {
      feed_ids.insert(var->id);
    }

    auto can_var_removed = [&](const Variable& var) { return !feed_ids.count(var->id) && !fetch_ids.count(var->id); };

    for (auto& group : identity_instr_groups_) {
      int i = 0;
      Variable first_input_var;
      std::vector<Variable> cannot_remove_vars;
      for (int instr_idx : group) {
        auto& instr = program[instr_idx];
        if (i == 0) {
          first_input_var = instr->inputs[0];
          if (!can_var_removed(instr->inputs[0])) {
            cannot_remove_vars.emplace_back(instr->inputs[0]);
          }
        }
        if (!can_var_removed(instr->outputs[0])) {
          cannot_remove_vars.emplace_back(instr->outputs[0]);
        }
        i++;
      }

      if (cannot_remove_vars.size() <= 1) {
        auto& reserved_var = cannot_remove_vars.size() == 1 ? cannot_remove_vars[0] : first_input_var;
        for (auto& instr_idx : group) {
          AddRemovedInstr(program, reserved_var, instr_idx);
        }
      } else {
        int j                      = 0;
        int num_cannot_remove_vars = cannot_remove_vars.size();
        for (auto& instr_idx : group) {
          auto& instr = program[instr_idx];
          // Reserve the front cannot_remove_vars.size - 1 identity instructions.
          if (j < num_cannot_remove_vars - 1) {
            origin2new_.emplace(instr->inputs[0].get(), cannot_remove_vars[j]);
            origin2new_.emplace(instr->outputs[0].get(), cannot_remove_vars[j + 1]);
          } else {
            Variable reserved_var = cannot_remove_vars[num_cannot_remove_vars - 1];
            AddRemovedInstr(program, reserved_var, instr_idx);
          }
          j++;
        }
      }
    }

    VLOG(4) << "origin2new_ {";
    for (auto& iter : origin2new_) {
      VLOG(4) << "  " << iter.first->id << " -> " << iter.second->id;
    }
    VLOG(4) << "}";
  }

  void AddRemovedInstr(const Program& program, const Variable& reserved_var, int instr_idx) {
    auto& instr = program[instr_idx];
    VLOG(3) << "Remove the " << instr_idx << "-th instruction: " << instr;
    remove_idxs_.insert(instr_idx);

    for (auto& var : std::vector<Variable>{instr->inputs[0], instr->outputs[0]}) {
      if (var->id != reserved_var->id) {
        origin2new_.emplace(var.get(), reserved_var);
      }
    }
  }

  void Clear() {
    remove_idxs_.clear();
    origin2new_.clear();
    identity_instr_groups_.clear();
  }

  std::unordered_set<int> remove_idxs_;
  std::unordered_map<_Variable_*, Variable> origin2new_;
  std::vector<std::set<int>> identity_instr_groups_;
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(RemoveIdentity) {
  CINN_REGISTER_PROGRAM_PASS(RemoveIdentity, cinn::frontend::pass::RemoveIdentityPass);

  return true;
}

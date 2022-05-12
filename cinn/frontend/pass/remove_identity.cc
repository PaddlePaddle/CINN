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
    CollectInfo(*program, fetch_ids);

    VLOG(5) << "Origin program: " << *program;
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
  }

 private:
  void CollectInfo(const Program& program, const std::unordered_set<std::string>& fetch_ids) {
    remove_idxs_.clear();
    origin2new_.clear();

    std::unordered_set<std::string> feed_ids;
    for (auto& var : program.GetInputs()) {
      feed_ids.insert(var->id);
    }
    for (int i = 0; i < program.size(); ++i) {
      const auto& instr = program[i];
      if (instr->op_type != "identity") {
        continue;
      }
      CHECK_EQ(instr->inputs.size(), 1) << "identity should have only 1 input.";
      CHECK_EQ(instr->outputs.size(), 1) << "identity should have only 1 output.";

      auto& input_var             = instr->inputs[0];
      auto& output_var            = instr->outputs[0];
      bool can_input_var_removed  = !feed_ids.count(input_var->id) && !fetch_ids.count(input_var->id);
      bool can_output_var_removed = !fetch_ids.count(output_var->id);
      if (can_input_var_removed || can_output_var_removed) {
        bool updated = false;
        if (can_input_var_removed) {
          updated = UpdateOrigin2New(input_var, output_var);
        }
        if (!updated && can_output_var_removed) {
          updated = UpdateOrigin2New(output_var, input_var);
        }
        if (updated) {
          VLOG(3) << "Remove the " << i << "-th instruction: " << instr;
          remove_idxs_.insert(i);
        }
      }
    }

    for (auto& v : origin2new_) {
      const auto& reserved_var = v.second;
      auto iter                = origin2new_.find(reserved_var.get());
      if (iter != origin2new_.end()) {
        VLOG(4) << "Update " << v.first->id << " -> " << reserved_var->id << " to " << v.first->id << " -> "
                << iter->second->id;
        origin2new_[v.first] = iter->second;
      }
    }

    VLOG(4) << "origin2new_ {";
    for (auto& iter : origin2new_) {
      VLOG(4) << "  " << iter.first->id << " -> " << iter.second->id;
    }
    VLOG(4) << "}";
  }

  bool UpdateOrigin2New(const Variable& origin, const Variable& new_var) {
    if (!origin2new_.count(origin.get())) {
      if (origin2new_.count(new_var.get())) {
        VLOG(4) << "Add " << origin->id << " -> " << origin2new_[new_var.get()]->id;
        origin2new_.emplace(origin.get(), origin2new_[new_var.get()]);
      } else {
        VLOG(4) << "Add " << origin->id << " -> " << new_var->id;
        origin2new_.emplace(origin.get(), new_var);
      }
      return true;
    }
    return false;
  }

  std::unordered_set<int> remove_idxs_;
  std::unordered_map<_Variable_*, Variable> origin2new_;
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(RemoveIdentity) {
  CINN_REGISTER_PROGRAM_PASS(RemoveIdentity, cinn::frontend::pass::RemoveIdentityPass);

  return true;
}

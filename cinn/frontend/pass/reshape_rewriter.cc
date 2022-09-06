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

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/program_pass.h"
#include "glog/logging.h"

namespace cinn {
namespace frontend {
namespace pass {

// ReshapeRewriterPass simplify instructions in following patterns:
//
// 1. Change reshape to identity, if the shape of input and output is the same. The identity can be removed by
// RemoveIdentityPass.
// 2. Simplify fill_constant + reshape to a single fill_constant, if varA/varB is not in fetch_ids.
//        fill_constant              fill_constant
//              | varA                     |
//           reshape           =>          | varA/varB
//              | varB                     |
//            instrX                     instrX
class ReshapeRewriterPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    CollectInfo(*program, fetch_ids);

    VLOG(3) << "Total remove " << remove_idxs_.size() << " instructions; replace " << replace_idxs_.size()
            << " instructions (reshape -> identity).";
    if (remove_idxs_.size() == 0 && replace_idxs_.size() == 0) {
      return;
    }

    NetBuilder builder("reshape_rewritter_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }

    for (int i = 0; i < program->size(); ++i) {
      const auto& instr = (*program)[i];
      if (remove_idxs_.count(i)) {
        // Change the attr shape of the previous fill_constant.
        auto iter         = outputs2instr_.find(instr->inputs[0].get());
        auto& input_instr = iter->second;
        CHECK(input_instr->op_type == "fill_constant") << "The previous op's type is not fill_constant, please check.";
        input_instr->outputs[0]     = instr->outputs[0];
        input_instr->attrs["shape"] = instr->outputs[0]->shape;
      } else {
        if (replace_idxs_.count(i)) {
          instr->op_type = "identity";
          instr->attrs.clear();
          instr->attrs_ordered.clear();
        }
        builder.AppendInstruction(instr);
      }
    }
    *program = builder.Build();
  }

 private:
  void CollectInfo(const Program& program, const std::unordered_set<std::string>& fetch_ids) {
    replace_idxs_.clear();
    remove_idxs_.clear();
    outputs2instr_.clear();

    std::unordered_map<_Variable_*, int> var_used_count;
    for (int i = 0; i < program.size(); ++i) {
      auto& instr = program[i];
      for (auto& var : instr->outputs) {
        outputs2instr_.emplace(var.get(), instr);
      }
      for (auto& var : instr->inputs) {
        var_used_count[var.get()]++;
      }
    }

    for (int i = 0; i < program.size(); ++i) {
      const auto& instr = program[i];
      if (instr->op_type != "reshape") {
        continue;
      }
      CHECK_EQ(instr->inputs.size(), 1) << "reshape should have only 1 input.";
      CHECK_EQ(instr->outputs.size(), 1) << "reshape should have only 1 output.";

      auto& input_var  = instr->inputs[0];
      auto& output_var = instr->outputs[0];

      bool matched = false;
      auto iter    = outputs2instr_.find(input_var.get());
      if (iter != outputs2instr_.end()) {
        auto& input_instr = iter->second;
        // Match the pattern is fill_constant -> reshape
        // reshape should be the only output instruction of fill_constant, and the output variable of fill_constant
        // cannot be in the fetch_ids.
        if (input_instr->op_type == "fill_constant" && var_used_count[input_var.get()] == 1 &&
            !fetch_ids.count(input_var->id)) {
          matched = true;
          remove_idxs_.insert(i);
          VLOG(3) << "Remove the " << i << "-th instruction: " << instr;
        }
      }
      if (!matched && (input_var->id != output_var->id) && (input_var->shape == output_var->shape)) {
        VLOG(3) << "Replace the " << i << "-th instruction to identity: " << instr;
        replace_idxs_.insert(i);
      }
    }
  }

  std::unordered_set<int> replace_idxs_;
  std::unordered_set<int> remove_idxs_;
  std::unordered_map<_Variable_*, Instruction> outputs2instr_;
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(ReshapeRewriter) {
  CINN_REGISTER_PROGRAM_PASS(ReshapeRewriter, cinn::frontend::pass::ReshapeRewriterPass);

  return true;
}

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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/program_pass.h"
#include "glog/logging.h"

namespace cinn {
namespace frontend {
namespace pass {

static std::unordered_map<std::string, std::function<void(const Instruction&, Instruction*)>> rewriter_ops = {
    {"reshape",
     [](const Instruction& fill_constant, Instruction* instr) -> void {
       (*instr)->op_type = "fill_constant";
       (*instr)->inputs.clear();
       // the outputs keep same

       CHECK((*instr)->attrs.count("shape")) << "The reshape op should has attribute [shape]!";
       auto new_shape           = (*instr)->attrs.at("shape");
       (*instr)->attrs          = fill_constant->attrs;
       (*instr)->attrs["shape"] = new_shape;
     }},
    {"scale", [](const Instruction& fill_constant, Instruction* instr) -> void {
       (*instr)->op_type = "fill_constant";
       (*instr)->inputs.clear();
       // the outputs keep same

       auto scale = (*instr)->attrs.count("scale") ? instr->GetAttrs<float>("scale") : 1.0f;
       auto bias  = (*instr)->attrs.count("bias") ? instr->GetAttrs<float>("bias") : 0.0f;
       auto bias_after_scale =
           (*instr)->attrs.count("bias_after_scale") ? instr->GetAttrs<bool>("bias_after_scale") : true;

       (*instr)->attrs = fill_constant->attrs;

       auto old_value = fill_constant.GetAttrs<float>("value");
       if (bias_after_scale) {
         (*instr)->attrs["value"] = scale * old_value + bias;
       } else {
         (*instr)->attrs["value"] = scale * (old_value + bias);
       }
     }}};

class FillConstantRewriterPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    VLOG(3) << "Before FillConstantRewriterPass:\n" << *program;
    auto input2instr = GetInput2Instr(program);

    std::unordered_set<const Instruction*> remove_instr;
    for (int i = 0; i < program->size(); ++i) {
      const auto& instr = (*program)[i];

      if (instr->op_type == "fill_constant") {
        RewriteFillConstant(instr, input2instr, fetch_ids, &remove_instr);
      }
    }
    VLOG(3) << "FillConstantRewriterPass Remove " << remove_instr.size() << " instruction";

    NetBuilder builder("reshape_rewritter_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }

    for (int i = 0; i < program->size(); ++i) {
      const auto& instr = (*program)[i];

      if (!remove_instr.count(&instr)) {
        builder.AppendInstruction(instr);
      }
    }
    *program = builder.Build();

    VLOG(3) << "After FillConstantRewriterPass:\n" << *program;
  }

 private:
  using Input2Instr = std::unordered_map<std::string, std::unordered_set<Instruction*>>;

  Input2Instr GetInput2Instr(Program* program) {
    Input2Instr input2instr;

    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& var : instr->inputs) {
        input2instr[var->id].insert(&instr);
      }
    }

    return input2instr;
  }

  void RewriteFillConstant(const Instruction& fill_constant,
                           const Input2Instr& input2instr,
                           const std::unordered_set<std::string>& fetch_ids,
                           std::unordered_set<const Instruction*>* remove_instr) {
    CHECK_EQ(fill_constant->op_type, std::string("fill_constant"));
    CHECK_EQ(fill_constant->outputs.size(), 1UL) << "The fill_constant op should just has one output! Please check.";
    const auto& out = fill_constant->outputs[0];

    if (!input2instr.count(out->id)) {
      // the fill constant's output is empty, skip
      return;
    }

    bool can_remove = true;
    for (auto* instr : input2instr.at(out->id)) {
      if (rewriter_ops.count((*instr)->op_type)) {
        rewriter_ops.at((*instr)->op_type)(fill_constant, instr);
        RewriteFillConstant(*instr, input2instr, fetch_ids, remove_instr);
      } else {
        can_remove = false;
      }
    }

    if (can_remove && !fetch_ids.count(out->id)) {
      remove_instr->insert(&fill_constant);
    }
  }
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(FillConstantRewriter) {
  CINN_REGISTER_PROGRAM_PASS(FillConstantRewriter, cinn::frontend::pass::FillConstantRewriterPass);

  return true;
}

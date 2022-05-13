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

#include <iostream>

#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/pattern.h"
#include "cinn/frontend/program_pass.h"
#include "glog/logging.h"

namespace cinn::frontend::pass {

class DotMergerPass : public ProgramPass {
 public:
  DotMergerPass(const std::string& name) : ProgramPass(name) { pattern_ = std::move(GeneratePattern()); }

 protected:
  void ApplyImpl(Program* prog,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    if (!Match(prog)) {
      return;
    }
    Rewrite(prog, fetch_ids, target);
  }

 private:
  Digraph GeneratePattern();
  bool Match(Program* prog);

  // TODO: More general rewrite logic.
  void Rewrite(Program* prog, const std::unordered_set<std::string>& fetch_ids, const common::Target& target);

 private:
  Digraph pattern_;
  Digraph program_;
  PatternMatcher matcher_;
  std::vector<PatternMatcher::pattern_map_t> matches_;
};

Digraph DotMergerPass::GeneratePattern() {
  PatternBuilder builder;
  auto* in_0     = builder.AddVar()->set_label("in_0");
  auto* in_1     = builder.AddVar()->set_label("in_1");
  auto* in_2     = builder.AddVar()->set_label("in_2");
  auto* out_0    = builder.AddVar()->set_label("out_0");
  auto* out_1    = builder.AddVar()->set_label("out_1");
  auto* matmul_0 = builder.AddInstr("matmul", std::vector<PatternVar*>{in_0, in_1}, std::vector<PatternVar*>{out_0})
                       ->set_label("matmul_0");
  auto* matmul_1 = builder.AddInstr("matmul", std::vector<PatternVar*>{in_0, in_2}, std::vector<PatternVar*>{out_1})
                       ->set_label("matmul_1");
  return builder.release();
};

bool DotMergerPass::Match(Program* prog) {
  program_ = std::move(ProgramGraphBuilder(*prog).release());
  PatternMatcher matcher;
  matcher.Init(pattern_, program_);
  matches_ = std::move(matcher.DetectPatterns());
  VLOG(5) << "matches size " << matches_.size();
  return matches_.size();
}

void DotMergerPass::Rewrite(Program* prog,
                            const std::unordered_set<std::string>& fetch_ids,
                            const common::Target& target) {
  for (const auto& match : matches_) {
    const Variable& in_0  = *GetMatchedVar(match, "in_0");
    const Variable& in_1  = *GetMatchedVar(match, "in_1");
    const Variable& in_2  = *GetMatchedVar(match, "in_2");
    const Variable& out_0 = *GetMatchedVar(match, "out_0");
    const Variable& out_1 = *GetMatchedVar(match, "out_1");

    // TODO: support more shapes.
    auto& in1_shape = in_1->shape;
    auto& in2_shape = in_2->shape;
    int axis        = 0;
    if (in_1->shape[0] == in_2->shape[0]) {
      axis = 1;
    }

    std::set<_Instruction_*> nodes_to_remove{GetMatchedInstr(match, "matmul_0")->get(),
                                             GetMatchedInstr(match, "matmul_1")->get()};

    NetBuilder builder("dot_merger_builder");
    size_t cnt{0};
    Variable slice0_out;
    Variable slice1_out;
    for (size_t i = 0; i < prog->size(); ++i) {
      auto& instr = (*prog)[i];
      if (nodes_to_remove.find(instr.get()) != nodes_to_remove.end()) {
        if (++cnt == nodes_to_remove.size()) {
          Variable matmul_out;
          auto concat_out = builder.Concat({in_1, in_2}, axis);
          if (axis == 1) {
            matmul_out = builder.Matmul(in_0, concat_out);
          } else {
            matmul_out = builder.Matmul(concat_out, in_0);
          }
          slice0_out = builder.Slice(matmul_out, {axis}, {0}, {in1_shape[axis]});
          slice1_out = builder.Slice(matmul_out, {axis}, {in1_shape[axis]}, {in1_shape[axis] + in2_shape[axis]});
        }
      } else {
        builder.AppendInstruction(instr);
      }
    }
    for (size_t i = 0; i < prog->size(); ++i) {
      auto& instr = (*prog)[i];
      for (auto& var : instr->inputs) {
        if (var.get() == out_0.get()) {
          var = slice0_out;
        }
        if (var.get() == out_1.get()) {
          var = slice1_out;
        }
      }
    }
    auto program = builder.Build();
    *prog        = std::move(program);
  }
}

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(DotMerger) {
  CINN_REGISTER_PROGRAM_PASS(DotMerger, ::cinn::frontend::pass::DotMergerPass);
  return true;
}

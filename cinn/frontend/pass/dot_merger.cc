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
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/pattern.h"
#include "cinn/frontend/program_pass.h"
#include "glog/logging.h"

namespace cinn::frontend::pass {

class DotMergerPass : public ProgramPass {
 public:
  DotMergerPass(const std::string& name) : ProgramPass(name) {}

 protected:
  void ApplyImpl(Program* prog,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    if (!Match(prog, fetch_ids, target)) {
      return;
    }
    Rewrite(prog, fetch_ids, target);
  }

 private:
  std::unique_ptr<Digraph> GeneratePattern(const std::unordered_set<std::string>& fetch_ids);
  bool Match(Program* prog, const std::unordered_set<std::string>& fetch_ids, const common::Target& target);

  // TODO: More general rewrite logic.
  void Rewrite(Program* prog, const std::unordered_set<std::string>& fetch_ids, const common::Target& target);

 private:
  std::unique_ptr<Digraph> pattern_;
  std::unique_ptr<Digraph> program_;
  PatternMatcher matcher_;
  std::vector<PatternMatcher::pattern_map_t> matches_;
};

template <typename T>
T GetAttr(const Instruction* instr, const char* attr, T default_value) {
  auto& attrs = instr->get()->attrs;
  if (attrs.count(attr)) {
    return absl::get<T>(attrs.at(attr));
  } else {
    return default_value;
  }
}

std::unique_ptr<Digraph> DotMergerPass::GeneratePattern(const std::unordered_set<std::string>& fetch_ids) {
  auto has_2d_shape = [](ProgramVar* var) -> bool { return var->raw()->get()->shape.size() == 2; };

  // TODO: move it into the base class
  auto not_fetch = [=](ProgramVar* var) -> bool {
    bool res = !fetch_ids.count(var->raw()->get()->id);
    return res;
  };
  auto in_matmul = [](ProgramVar* var) -> bool {
    for (auto target : var->prog()->adj().GetTargets(var)) {
      auto* prog = dynamic_cast<ProgramInstr*>(target.end());
      if (prog && prog->raw()->get()->op_type == "matmul" && !GetAttr(prog->raw(), "trans_a", false) &&
          !GetAttr(prog->raw(), "trans_b", false)) {
        return true;
      }
    }
    return false;
  };
  auto out_matmul = [](ProgramVar* var) -> bool {
    bool res = false;
    for (auto edge : var->prog()->adj().edges()) {
      auto* first = dynamic_cast<ProgramInstr*>(edge.first);
      if (first && first->raw()->get()->op_type == "matmul" && edge.second == var) {
        return true;
      }
    }
    return false;
  };
  auto no_trans = [](ProgramInstr* instr) -> bool {
    bool res = false;
    return !GetAttr(instr->raw(), "trans_a", false) && !GetAttr(instr->raw(), "trans_b", false);
  };

  PatternBuilder builder;
  auto* in_0     = builder.AddVar()->Assert(has_2d_shape)->Assert(in_matmul)->set_label("in_0");
  auto* in_1     = builder.AddVar()->Assert(has_2d_shape)->Assert(in_matmul)->set_label("in_1");
  auto* in_2     = builder.AddVar()->Assert(has_2d_shape)->Assert(in_matmul)->set_label("in_2");
  auto* out_0    = builder.AddVar()->Assert(has_2d_shape)->Assert(out_matmul)->Assert(not_fetch)->set_label("out_0");
  auto* out_1    = builder.AddVar()->Assert(has_2d_shape)->Assert(out_matmul)->Assert(not_fetch)->set_label("out_1");
  auto* matmul_0 = builder.AddInstr("matmul", std::vector<PatternVar*>{in_0, in_1}, std::vector<PatternVar*>{out_0})
                       ->set_label("matmul_0")
                       ->Assert(no_trans);
  auto* matmul_1 = builder.AddInstr("matmul", std::vector<PatternVar*>{in_0, in_2}, std::vector<PatternVar*>{out_1})
                       ->set_label("matmul_1")
                       ->Assert(no_trans);
  return builder.release();
};

bool DotMergerPass::Match(Program* prog,
                          const std::unordered_set<std::string>& fetch_ids,
                          const common::Target& target) {
  program_ = std::move(ProgramGraphBuilder(*prog).release());
  PatternMatcher matcher;
  pattern_ = std::move(GeneratePattern(fetch_ids));
  matcher.Init(*pattern_, *program_);
  matches_ = std::move(matcher.DetectPatterns());
  VLOG(5) << "matches size " << matches_.size();
  return matches_.size();
}

int in_idx(const Instruction& instr, const Variable& var) {
  int res = -1;
  for (size_t i = 0; i < instr->inputs.size(); ++i) {
    if (instr->inputs[i].get() == var.get()) {
      res = i;
    }
  }
  return res;
}

void DotMergerPass::Rewrite(Program* prog,
                            const std::unordered_set<std::string>& fetch_ids,
                            const common::Target& target) {
  for (const auto& match : matches_) {
    const auto& in0     = *get_mapped_var(match, "in_0")->raw();
    const auto& in1     = *get_mapped_var(match, "in_1")->raw();
    const auto& in2     = *get_mapped_var(match, "in_2")->raw();
    const auto& out0    = *get_mapped_var(match, "out_0")->raw();
    const auto& out1    = *get_mapped_var(match, "out_1")->raw();
    const auto& matmul0 = *get_mapped_instr(match, "matmul_0")->raw();
    const auto& matmul1 = *get_mapped_instr(match, "matmul_1")->raw();

    DepthFirstSearch dfs(*program_);
    if (dfs.accessible(get_mapped_var(match, "out_0"), get_mapped_var(match, "in_2")) ||
        dfs.accessible(get_mapped_var(match, "out_1"), get_mapped_var(match, "in_1"))) {
      continue;
    }

    bool lhs{true};
    int axis{1};
    if (in_idx(matmul0, in0) != in_idx(matmul1, in0) || in_idx(matmul0, in1) != in_idx(matmul1, in2)) {
      continue;
    } else if (in_idx(matmul0, in0) == 1) {
      lhs  = false;
      axis = 0;
    }

    std::set<_Instruction_*> nodes_to_remove{matmul0.get(), matmul1.get()};

    NetBuilder builder("dot_merger_builder");
    Variable slice0_out;
    Variable slice1_out;
    int cnt{0};
    for (size_t i = 0; i < prog->size(); ++i) {
      auto& instr = (*prog)[i];
      if (nodes_to_remove.find(instr.get()) != nodes_to_remove.end()) {
        if (++cnt == nodes_to_remove.size()) {
          Variable matmul_out;
          auto concat_out = builder.Concat({in1, in2}, axis);
          if (!lhs) {
            matmul_out = builder.Matmul(concat_out, in0);
          } else {
            matmul_out = builder.Matmul(in0, concat_out);
          }
          slice0_out = builder.Slice(matmul_out, {axis}, {0}, {in1->shape[axis]});
          slice1_out = builder.Slice(matmul_out, {axis}, {in1->shape[axis]}, {in1->shape[axis] + in2->shape[axis]});
        }
      } else {
        builder.AppendInstruction(instr);
      }
    }
    for (size_t i = 0; i < prog->size(); ++i) {
      auto& instr = (*prog)[i];
      for (auto& var : instr->inputs) {
        if (var.get() == out0.get()) {
          var = slice0_out;
        }
        if (var.get() == out1.get()) {
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

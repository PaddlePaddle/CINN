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

#include <ios>
#include <unordered_map>
#include <unordered_set>

#include "absl/status/statusor.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/program_pass.h"
#include "glog/logging.h"

namespace cinn {
namespace frontend {
namespace pass {

class GemmRewriterPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

  void ApplyImpl(Program* prog, const std::unordered_set<std::string>& fetch_ids, const common::Target& target) {
    if (target.arch != Target::Arch::NVGPU || !prog->size()) {
      return;
    }

    LOG(INFO) << "-- Origin: " << *prog;

    CollectInfo(*prog);

    std::vector<Instruction> instrs;
    for (int i = prog->size() - 1; i >= 0; i--) {
      auto& instr = prog->operator[](i);
      if (instr->op_type == "elementwise_add") {
        auto may_fused = TryGetMulBias(instr, fetch_ids);
        if (may_fused) {
          instrs.emplace_back(*may_fused);
          continue;
        }
      }
      if (!removed_instrs_.count(instr.get())) {
        instrs.emplace_back(instr);
      }
    }

    NetBuilder builder("gemm_rewriter_builder");
    for (auto& var : prog->GetInputs()) {
      builder.CreateInput(var);
    }
    for (auto it = instrs.rbegin(); it != instrs.rend(); it++) {
      builder.AppendInstruction(*it);
    }

    *prog = builder.Build();

    for (size_t i = 0; i < prog->size(); i++) {
      auto& inputs = (*prog)[i]->inputs;
      for (size_t j = 0; j < inputs.size(); j++) {
        if (origin2new_.count(inputs[j].get())) {
          inputs[j] = origin2new_.at(inputs[j].get());
        }
      }
    }
    LOG(INFO) << "-- Update: " << *prog;
  }

 private:
  void CollectInfo(const Program& prog) {
    for (size_t i = 0; i < prog.size(); i++) {
      auto& instr = prog[i];
      for (auto& var : instr->outputs) {
        output2instr_.emplace(var.get(), instr);
      }
      for (auto& var : instr->inputs) {
        var_used_count_[var.get()]++;
      }
    }
  }

  absl::optional<Instruction> TryGetMulBias(Instruction instr, const std::unordered_set<std::string>& fetch_ids) {
    CHECK_EQ(instr->inputs.size(), 2) << "elementwise should have only two inputs";
    std::vector<Variable> inputs;
    bool trans_a = false;
    bool trans_b = false;
    for (auto& var : instr->inputs) {
      auto it = output2instr_.find(var.get());
      if (it != output2instr_.end() && it->second->op_type == "matmul") {
        // If the output var of matmul is consumed by more than one instruction or
        // a fetch var, just skip to fuse it.
        CHECK_GT(var_used_count_.count(var.get()), 0);
        if ((var_used_count_.at(var.get()) > 1) || fetch_ids.count(var->id)) {
          continue;
        }

        auto& matmul_instr = it->second;
        // set inputs of mulbias
        inputs     = matmul_instr->inputs;
        auto& bias = instr->inputs[0].get() == var.get() ? instr->inputs[1] : instr->inputs[0];
        inputs.emplace_back(bias);
        // set attrs of mulbias
        auto& attrs = matmul_instr->attrs;
        if (attrs.count("trans_a")) {
          trans_a = absl::get<bool>(attrs.at("trans_a"));
        }
        if (attrs.count("trans_b")) {
          trans_b = absl::get<bool>(attrs.at("trans_b"));
        }

        // After the fusion, matmul and elementwise_add should be removed.
        removed_instrs_.emplace(matmul_instr.get());
        removed_instrs_.emplace(instr.get());
        break;
      }
    }

    if (inputs.size() == 3) {
      NetBuilder builder("create_mulbias");
      for (auto& var : inputs) {
        builder.CreateInput(var);
      }

      LOG(INFO) << "-- trans_a = " << std::boolalpha << trans_a;
      LOG(INFO) << "-- trans_b = " << std::boolalpha << trans_b;
      auto new_out = builder.MulBias(inputs[0], inputs[1], inputs[2], 1, 1, trans_a, trans_b);
      origin2new_.emplace(instr.GetOutput(0).get(), new_out);
      return builder.Build()[0];
    }

    CHECK_EQ(inputs.size(), 0);
    return absl::nullopt;
  }

  std::unordered_set<_Instruction_*> removed_instrs_;
  std::unordered_map<_Variable_*, Variable> origin2new_;
  std::unordered_map<_Variable_*, Instruction> output2instr_;
  std::unordered_map<_Variable_*, int> var_used_count_;
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

namespace fp = ::cinn::frontend::pass;
CINN_REGISTER_HELPER(GemmRewriter) {
  CINN_REGISTER_PROGRAM_PASS(GemmRewriter, fp::GemmRewriterPass);

  return true;
}

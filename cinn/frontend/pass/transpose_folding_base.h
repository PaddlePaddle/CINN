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

#include <sstream>
#include <string>
#include <unordered_set>

#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

class TransposeFoldingBase : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;
  using In2InstrType  = absl::flat_hash_map<std::string, std::unordered_set<Instruction*>>;
  using Out2InstrType = absl::flat_hash_map<std::string, Instruction*>;

 protected:
  virtual void set_target_instrs() = 0;

  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    VLOG(4) << "-- Before folding: " << *program;
    set_target_instrs();
    // `out2instr` is used to represent the mapping of Output to Instruction.
    Out2InstrType out2instr;
    // `in2instr` is used to represent the mapping of Input to Instruction.
    In2InstrType in2instr;
    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& out : instr->outputs) {
        out2instr[out->id] = &instr;
      }
      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
    }

    // `remove_instrs` is used to represent Instructions of which type is transpose to be deleted.
    absl::flat_hash_set<Instruction*> remove_instrs;
    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      if (target_instrs_.count(instr->op_type)) {
        DoMatmulFoldOptimize(&instr, out2instr, in2instr, fetch_ids, &remove_instrs);
      }
    }

    NetBuilder builder("transpose_folding_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (!remove_instrs.count(&(*program)[i])) {
        builder.AppendInstruction((*program)[i]);
      }
    }
    *program = builder.Build();
    VLOG(4) << "-- After folding: " << *program;
  }

  // get can fold instruction in order, more front, more near from dot op
  // the `instr` param is the next instruction of matmul, not the matmul
  std::vector<Instruction*> GetFoldInstruction(Instruction* instr,
                                               const Out2InstrType& out2instr,
                                               const In2InstrType& in2instr,
                                               bool from_input) const {
    if (!fold_instrs_.count((*instr)->op_type)) {
      return {};
    }
    CHECK_EQ((*instr)->inputs.size(), 1UL) << "The op " << (*instr)->op_type << " should has 1 input.";
    CHECK_EQ((*instr)->outputs.size(), 1UL) << "The op " << (*instr)->op_type << " should has 1 output.";

    if (!from_input && in2instr.at((*instr)->inputs[0]->id).size() != 1) {
      // the matmul's output should only link to one op
      return {};
    }

    std::vector<Instruction*> res           = {instr};
    std::unordered_set<std::string> visited = {(*instr)->op_type};

    auto cur_instr = instr;
    while (cur_instr) {
      Instruction* next_instr = nullptr;

      if (from_input) {
        // scale -> transpose -> matmul ==> {"transpose", "scale"}
        auto iter = out2instr.find((*cur_instr)->inputs[0]->id);
        if (iter != out2instr.end()) {
          next_instr = iter->second;
        }
      } else {
        // matmul -> transpose -> scale ==> {"transpose", "scale"}
        auto iter = in2instr.find((*cur_instr)->outputs[0]->id);
        if ((iter != in2instr.end()) && iter->second.size() == 1) {
          next_instr = *iter->second.begin();
        }
      }

      if (next_instr && fold_instrs_.count((*next_instr)->op_type) && !visited.count((*next_instr)->op_type)) {
        // found can fold instruction and not repeat
        res.emplace_back(next_instr);
        visited.emplace((*next_instr)->op_type);
      } else {
        // the fold instructions must consecutive
        break;
      }

      cur_instr = next_instr;
    }

    return res;
  }

  bool IsValidTranspose(const Instruction& transpose) const {
    if ("transpose" != transpose->op_type) {
      return false;
    }

    // `axis` of tranpose must be consecutive in the reverse order,
    // excluding the first dim
    auto axis = transpose.GetAttrs<std::vector<int>>("axis");
    if (axis.size() == 3) {
      // In the batched martix multiplication, the first dim should be batch dim.
      if (axis[0] != 0) {
        return false;
      }
      for (size_t i = 1; i < axis.size(); ++i) {
        if (axis[i] != axis.size() - i) {
          return false;
        }
      }
    } else if (axis.size() == 2) {
      // In the normal martix multiplication, the axis should be consecutive in the reverse order.
      for (size_t i = 1; i < axis.size(); ++i) {
        if (axis[i] != axis.size() - 1 - i) {
          return false;
        }
      }
    } else {
      // Otherwise, cannot folding
      return false;
    }

    return true;
  }

  bool IsValidScale(const Instruction& scale) const {
    if ("scale" != scale->op_type) {
      return false;
    }

    float bias = scale->attrs.count("bias") ? absl::get<float>(scale->attrs.at("bias")) : 0.0f;
    return (bias == 0.0f);
  }

  virtual void DoMatmulFoldOptimize(Instruction* instr,
                                    const Out2InstrType& out2instr,
                                    const In2InstrType& in2instr,
                                    const std::unordered_set<std::string>& fetch_ids,
                                    absl::flat_hash_set<Instruction*>* remove_instrs) const = 0;

  std::unordered_set<std::string> target_instrs_;
  std::unordered_set<std::string> fold_instrs_{"transpose", "scale"};
};

}  // namespace cinn::frontend::pass

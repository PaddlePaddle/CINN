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

#include <string>
#include <unordered_set>

#include "cinn/common/target.h"
#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/utils/type_defs.h"

namespace cinn::frontend::pass {

using cinn::utils::DimType;
using cinn::utils::ShapeType;

using InputToOpMap = std::unordered_map<std::string, std::unordered_set<Instruction*>>;

class FillConstantKey {
 public:
  FillConstantKey(const ShapeType& shape, float value, const std::string& dtype, bool force_cpu) {
    SetKey(shape, value, dtype, force_cpu);
  }

  void SetKey(const ShapeType& shape, float value, const std::string& dtype, bool force_cpu) {
    shape_     = shape;
    value_     = value;
    force_cpu_ = force_cpu;
    dtype_     = dtype;
  }

  bool operator==(const FillConstantKey& other) const {
    return std::equal(shape_.begin(), shape_.end(), other.shape_.begin()) && value_ == other.value_ &&
           force_cpu_ == other.force_cpu_ && dtype_ == other.dtype_;
  }
  bool operator!=(const FillConstantKey& other) const { return !this->operator==(other); }

  struct Hash {
    static size_t hash_combine(size_t seed, size_t value) {
      return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }

    size_t operator()(const FillConstantKey& key) const {
      size_t ret = 0;

      std::hash<DimType> dim_hasher;
      std::for_each(
          key.shape_.begin(), key.shape_.end(), [&](const DimType& dim) { ret = hash_combine(ret, dim_hasher(dim)); });

      ret = hash_combine(ret, std::hash<float>()(key.value_));
      ret = hash_combine(ret, std::hash<bool>()(key.force_cpu_));
      ret = hash_combine(ret, std::hash<std::string>()(key.dtype_));

      return ret;
    }
  };

 private:
  bool ShapeEqual(const ShapeType& shape1, const ShapeType& shape2) {
    return std::equal(shape1.begin(), shape1.end(), shape2.begin());
  }

  ShapeType shape_;
  float value_;
  bool force_cpu_;
  std::string dtype_;
};

// Pass `FillConstantFolding` folds fill_constant into dot, then both of them can be implemented by a
// GEMM kernel. For each dot operator, try folding every input that belong output of fill_constant.
// If output of tranpose in `fetch_ids`, keep the operator.
class FillConstantFoldingPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) const override {
    auto in2instr = GetInputToOpMap(program);

    // `fill_constant_set` is used to represent the first fill_constant and its output variable
    std::unordered_map<FillConstantKey, Variable*, FillConstantKey::Hash> fill_constant_set;
    // `remove_instrs` is used to represent Instructions which type is fill_constant to be deleted.
    std::unordered_set<Instruction*> remove_instrs;

    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];

      if ("fill_constant" != instr->op_type) {
        // not fill_constant op, skip
        continue;
      }

      CHECK_EQ(instr->outputs.size(), 1UL)
          << "The fill_constant op should has one, and only one output ! Please check.";
      if (fetch_ids.find(instr->outputs[0]->id) != fetch_ids.end()) {
        // the fill constant's output variable was fetched, skip
        VLOG(4) << "Cannot remove fill_constant, because Var [" << instr->outputs[0]->id
                << "] was fetched by other op ! ";
        continue;
      }

      const auto& shape = instr.GetAttrs<ShapeType>("shape");
      auto value        = instr.GetAttrs<float>("value");
      const auto& dtype = instr.GetAttrs<std::string>("dtype");
      auto force_cpu    = instr.GetAttrs<bool>("force_cpu");

      FillConstantKey key(shape, value, dtype, force_cpu);
      if (!fill_constant_set.count(key)) {
        VLOG(4) << "The fill_constant, whose output is Var [" << instr->outputs[0]->id
                << "], cannot remove because it is the first fill_costant ! ";
        // retain the first fill constant op node
        fill_constant_set.emplace(key, &instr->outputs[0]);
        continue;
      }

      VLOG(4) << "Try remove fill_constant, whose output is Var [" << instr->outputs[0]->id << "]. ";
      remove_instrs.insert(&instr);

      auto constant_name = instr->outputs[0]->id;
      ReLinkFillConstant(in2instr, constant_name, fill_constant_set.at(key));
    }

    CinnBuilder builder("fill_constant_folding_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (remove_instrs.end() != remove_instrs.find(&(*program)[i])) continue;
      builder.AppendInstruction((*program)[i]);
    }
    *program = builder.Build();
  }

 private:
  static InputToOpMap GetInputToOpMap(Program* program) {
    // `in2instr` is used to represent the mapping of Input to Instruction.
    InputToOpMap in2instr;

    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];

      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
    }
    return in2instr;
  }

  static void ReLinkFillConstant(const InputToOpMap& in2instr, const std::string& input_var_name, Variable* out_var) {
    if (in2instr.count(input_var_name)) {
      LOG(WARNING) << "Var [" << input_var_name << "] not used by other op ! ";
      return;
    }

    VLOG(4) << "Try replace the input Var [" << input_var_name << "] to [" << (*out_var)->id
            << "], because the fill_constant will be folding.";

    const auto& output_ops = in2instr.at(input_var_name);
    for (auto op : output_ops) {
      auto find_input = [&](const std::string& input_name) {
        return std::find_if(
            (*op)->inputs.begin(), (*op)->inputs.end(), [&](const Variable& var) { return var->id == input_name; });
      };

      auto it = find_input(input_var_name);
      // Why Loop : To avoid the op's inputs are the same variable !
      while (it != (*op)->inputs.end()) {
        (*op)->inputs.erase(it);
        (*op)->inputs.emplace_back(*out_var);

        it = find_input(input_var_name);
      }
    }
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(FillConstantFolding) {
  CINN_REGISTER_PROGRAM_PASS(FillConstantFolding, ::cinn::frontend::pass::FillConstantFoldingPass);

  return true;
}

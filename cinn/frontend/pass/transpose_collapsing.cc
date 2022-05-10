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
#include <unordered_map>
#include <unordered_set>

#include "cinn/common/target.h"
#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

using cinn::utils::DimType;
using cinn::utils::ShapeType;

using OutputToOpMap = std::unordered_map<std::string, Instruction*>;
using InputToOpMap  = std::unordered_map<std::string, std::unordered_set<Instruction*>>;

// Pass `TransposeCollapsing` folds multi transpose into one.
class TransposeCollapsingPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) const override {
    VLOG(4) << "Run TransposeCollapsingPass";
    // `out2instr` is used to represent the mapping of Output to Instruction.
    OutputToOpMap out2instr;
    // `in2instr` is used to represent the mapping of Input to Instruction.
    InputToOpMap in2instr;
    // all transpose op in program
    std::unordered_set<Instruction*> all_transpose;

    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& out : instr->outputs) {
        out2instr[out->id] = &instr;
      }
      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
      if ("transpose" == instr->op_type) {
        all_transpose.insert(&instr);
      }
    }

    // the useless transpose op need to remove from program
    std::unordered_set<Instruction*> remove_instrs;
    FoldingTransposeVertical(all_transpose, fetch_ids, in2instr, out2instr, &remove_instrs);

    for (auto instr : remove_instrs) {
      if (all_transpose.count(instr)) {
        all_transpose.erase(instr);
      }
    }
    FoldingTransposeHorizontal(all_transpose, fetch_ids, in2instr, out2instr, &remove_instrs);

    CinnBuilder builder("transpose_collapsing_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (remove_instrs.end() == remove_instrs.find(&(*program)[i])) {
        builder.AppendInstruction((*program)[i]);
      }
    }
    *program = builder.Build();
  }

 private:
  static void FoldingTransposeVertical(const std::unordered_set<Instruction*>& all_transpose,
                                       const std::unordered_set<std::string>& fetch_ids,
                                       const InputToOpMap& in2instr,
                                       const OutputToOpMap& out2instr,
                                       std::unordered_set<Instruction*>* remove_instrs) {
    // the transpose op should not remove
    std::unordered_set<Instruction*> visited_instrs;
    for (auto transpose : all_transpose) {
      if (remove_instrs->count(transpose) || visited_instrs.count(transpose)) {
        // the transpose op had been fused, skip
        continue;
      }

      // Fuse transpose from front to back, the fuse path is unique
      auto first_transpose = FindFirstTranspose(transpose, out2instr);
      TryFuseTranspose(first_transpose, fetch_ids, in2instr, remove_instrs, &visited_instrs);
    }
  }

  static Instruction* FindFirstTranspose(Instruction* transpose, const OutputToOpMap& out2instr) {
    auto first_transpose = transpose;

    auto input_name = (*first_transpose)->inputs.front()->id;
    // Q: Why check whether transpose's input in out2instr ?
    // A: The input may from fetch_ids other than another op's output.
    //    Obviously, the transpose op is the first transpose in the situation.
    while (out2instr.count(input_name)) {
      auto instr = out2instr.at(input_name);
      if ("transpose" != (*instr)->op_type) {
        // if input of transpose is not output of another transpose, it is the first transpose.
        break;
      }

      input_name      = (*instr)->inputs.front()->id;
      first_transpose = instr;
    }
    return first_transpose;
  }

  static void TryFuseTranspose(Instruction* transpose,
                               const std::unordered_set<std::string>& fetch_ids,
                               const InputToOpMap& in2instr,
                               std::unordered_set<Instruction*>* remove_instrs,
                               std::unordered_set<Instruction*>* visited_instrs) {
    visited_instrs->insert(transpose);

    const auto& input      = (*transpose)->inputs.front();
    const auto& input_name = input->id;

    const auto& output      = (*transpose)->outputs.front();
    const auto& output_name = output->id;

    const auto& axis = transpose->GetAttrs<ShapeType>("axis");
    CHECK_EQ(axis.size(), input->shape.size())
        << "The transpose's axis size should equal with input variable's shape size, but the transpose of ["
        << input->id << "] not ! Please check.";

    if (CheckTransposeBorder(transpose, in2instr)) {
      VLOG(4) << "The transpose op {input[" << input_name << "], output[" << output_name << "], axis["
              << cinn::utils::Join(axis, ",") << "]} is a output op of graph, connot fuse, skip.";
      return;
    }

    // CheckTransposeBorder ensure `output_name` existed in `in2instr`
    const auto& out_instrs = in2instr.at(output_name);
    if (CheckTransposeUseless(axis)) {
      VLOG(4) << "The transpose op {input[" << input_name << "], output[" << output_name << "], axis["
              << cinn::utils::Join(axis, ",") << "]} is useless, remove.";
      for (auto instr : out_instrs) {
        // replace the input to transpose's input
        ReplaceInputVariable(instr, output_name, input);
      }
      remove_instrs->insert(transpose);

      for (auto instr : out_instrs) {
        if ("transpose" == (*instr)->op_type) {
          // if the next instruction is transpose op, continue fuse
          TryFuseTranspose(instr, fetch_ids, in2instr, remove_instrs, visited_instrs);
        }
      }
      return;
    }

    if (!CheckOutputContainTranspose(transpose, in2instr)) {
      VLOG(4) << "The transpose op {input[" << input_name << "], output[" << output_name << "], axis["
              << cinn::utils::Join(axis, ",") << "]} doesn't has output link to transpose, skip.";
      return;
    }

    bool can_remove = true;
    std::unordered_set<Instruction*> next_fused_instrs;

    for (auto instr : out_instrs) {
      if ("transpose" != (*instr)->op_type) {
        // the transpose was used by other non-transpose op, cannot remove, skip
        can_remove = false;
        continue;
      }

      const auto& next_axis = instr->GetAttrs<ShapeType>("axis");
      // we can fuse two transpose by fuse the two axes like:
      // step |    axis   | after_transpose
      //  1   | [0, 2, 1] | [0, 2, 1]
      //  2   | [2, 1, 0] | [1, 2, 0]
      // so we can fuse tranpose([0, 2, 1]) and tranpose([2, 1, 0]) into tranpose([1, 2, 0])
      const auto& fused_axis = FuseTransposeAxis(axis, next_axis);
      auto fused_transpose   = FuseTransposeImpl(transpose, instr, fused_axis);

      VLOG(4) << "Fuse transpose of {input[" << input_name << "], output[" << output_name << "], axis ["
              << cinn::utils::Join(axis, ",") << "]} and transpose of {input[" << (*instr)->inputs.front()->id
              << "], output[" << (*instr)->outputs.front()->id << "], axis [" << cinn::utils::Join(next_axis, ",")
              << "]} into transpose of {input[" << input_name << "], output[" << (*instr)->outputs.front()->id
              << "], axis[" << cinn::utils::Join(fused_axis, ",") << "]}.";

      next_fused_instrs.insert(fused_transpose);
    }

    if (can_remove && !fetch_ids.count(output_name)) {
      VLOG(4) << "Remove transpose of {input[" << input_name << "], output[" << output_name << "], axis ["
              << cinn::utils::Join(axis, ",") << "]}.";
      remove_instrs->insert(transpose);
    }

    for (auto instr : next_fused_instrs) {
      TryFuseTranspose(instr, fetch_ids, in2instr, remove_instrs, visited_instrs);
    }
  }

  static bool CheckTransposeBorder(Instruction* transpose, const InputToOpMap& in2instr) {
    const auto& output_name = (*transpose)->outputs.front()->id;
    return !in2instr.count(output_name) || in2instr.at(output_name).empty();
  }

  static bool CheckOutputContainTranspose(Instruction* transpose, const InputToOpMap& in2instr) {
    const auto& output_name = (*transpose)->outputs.front()->id;
    for (auto instr : in2instr.at(output_name)) {
      if ("transpose" == (*instr)->op_type) {
        return true;
      }
    }
    // the first transpose's output is not anyone transpose's input
    return false;
  }

  static bool CheckTransposeUseless(const ShapeType& axis) {
    // if the transpose axis like {0, 1, 2, 3, 4, 5}, the transpose is useless, remove
    if (axis.front() != 0) return false;
    return std::is_sorted(axis.begin(), axis.end(), [](DimType dim1, DimType dim2) { return dim1 + 1 == dim2; });
  }

  static void ReplaceInputVariable(Instruction* op, const std::string& old_input_name, const Variable& new_input) {
    auto find_input = [&](const std::string& input_name) {
      return std::find_if(
          (*op)->inputs.begin(), (*op)->inputs.end(), [&](const Variable& v) { return input_name == v->id; });
    };

    auto it = find_input(old_input_name);
    // Why Loop : To avoid the op's inputs are the same variable !
    while (it != (*op)->inputs.end()) {
      // erase previous fill_constant output var and replace to new fill_constant output var
      auto next_it = (*op)->inputs.erase(it);
      // keep the input place same, it's very important
      (*op)->inputs.insert(next_it, new_input);
      // try find next var if there exist same input
      it = find_input(old_input_name);
    }
  }

  static ShapeType FuseTransposeAxis(const ShapeType& old_axis, const ShapeType& new_axis) {
    CHECK_EQ(old_axis.size(), new_axis.size())
        << "The transpose axis size should be " << old_axis.size() << ", but here " << new_axis.size();

    ShapeType axis = old_axis;
    for (int i = 0; i < new_axis.size(); ++i) {
      axis[i] = old_axis[new_axis[i]];
    }
    return axis;
  }

  static Instruction* FuseTransposeImpl(Instruction* transpose1, Instruction* transpose2, const ShapeType& fused_axis) {
    (*transpose2)->inputs.front() = (*transpose1)->inputs.front();
    transpose2->SetAttr("axis", fused_axis);
    return transpose2;
  }

  class TransposeKey {
   public:
    TransposeKey(const std::string& input_id, const ShapeType& axis) { SetKey(input_id, axis); }

    void SetKey(const std::string& input_id, const ShapeType& axis) {
      input_id_ = input_id;
      axis_     = axis;
    }

    bool operator==(const TransposeKey& other) const {
      return std::equal(axis_.begin(), axis_.end(), other.axis_.begin()) && input_id_ == other.input_id_;
    }
    bool operator!=(const TransposeKey& other) const { return !this->operator==(other); }

    struct Hash {
      size_t operator()(const TransposeKey& key) const {
        std::string ret;

        ret.append(key.input_id_);
        std::for_each(key.axis_.begin(), key.axis_.end(), [&](const DimType& dim) { ret.append(std::to_string(dim)); });

        return std::hash<std::string>()(ret);
      }
    };

   private:
    std::string input_id_;
    ShapeType axis_;
  };

  static void FoldingTransposeHorizontal(const std::unordered_set<Instruction*>& all_transpose,
                                         const std::unordered_set<std::string>& fetch_ids,
                                         const InputToOpMap& in2instr,
                                         const OutputToOpMap& out2instr,
                                         std::unordered_set<Instruction*>* remove_instrs) {
    std::unordered_map<TransposeKey, Variable*, TransposeKey::Hash> transpose_set;
    for (auto transpose : all_transpose) {
      if (remove_instrs->count(transpose)) {
        continue;
      }

      const auto& input_id  = (*transpose)->inputs.front()->id;
      const auto& output_id = (*transpose)->outputs.front()->id;
      const auto& axis      = transpose->GetAttrs<ShapeType>("axis");

      TransposeKey key(input_id, axis);
      if (!transpose_set.count(key)) {
        VLOG(4) << "The transpose, whose output [" << output_id
                << "], cannot remove because it is the first transpose ! ";
        transpose_set.emplace(key, &(*transpose)->outputs.front());
        continue;
      }

      if (fetch_ids.find(output_id) != fetch_ids.end()) {
        // the fill constant's output variable was fetched, skip
        VLOG(4) << "Cannot remove transpose, because the output [" << output_id << "] was fetched by other op ! ";
        continue;
      }

      VLOG(4) << "Try remove transpose, whose output [" << output_id << "]. ";
      remove_instrs->insert(transpose);

      const auto& output_ops = in2instr.at(output_id);
      for (auto op : output_ops) {
        ReplaceInputVariable(op, output_id, *transpose_set.at(key));
      }
    }
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(TransposeCollapsing) {
  CINN_REGISTER_PROGRAM_PASS(TransposeCollapsing, ::cinn::frontend::pass::TransposeCollapsingPass);

  return true;
}

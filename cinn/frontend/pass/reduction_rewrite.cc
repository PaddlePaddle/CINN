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
#include "cinn/frontend/program_pass.h"

namespace cinn::frontend::pass {

class ReductionRewritePass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;
  void ApplyImpl(Program* program, const Target& target) const;

 private:
  bool IsDimConsecutive(std::vector<int> dim) const;
  std::array<int, 3> DivideShapeByDim(const std::vector<int>& shape, const std::vector<int>& dim) const;
  bool IsHighPerformance(bool on_row, const std::array<int, 3>& dim) const;
  bool CanRewriteReduction(const Instruction& instr) const;
  Instruction RewriteReduction(const Instruction& instr) const;
};

void ReductionRewritePass::ApplyImpl(Program* program, const Target& target) const {
  CinnBuilder builder("reduction_rewrite_builder");
  for (auto& var : program->GetInputs()) {
    builder.CreateInput(var);
  }
  for (size_t i = 0; i < program->size(); ++i) {
    const auto& instr = (*program)[i];
    if (CanRewriteReduction(instr)) {
      VLOG(1) << "Rewrite instruction:\n" << instr;
      builder.AppendInstruction(RewriteReduction(instr));
    } else {
      builder.AppendInstruction(instr);
    }
  }
  *program = builder.Build();
}

bool ReductionRewritePass::IsDimConsecutive(std::vector<int> dim) const {
  std::sort(dim.begin(), dim.end());
  for (size_t i = 1; i < dim.size(); ++i) {
    if (1 != dim[i] - dim[i - 1]) return false;
  }
  return true;
}

std::array<int, 3> ReductionRewritePass::DivideShapeByDim(const std::vector<int>& shape,
                                                          const std::vector<int>& dim) const {
  constexpr size_t maj_idx   = 0;
  constexpr size_t mid_idx   = 1;
  constexpr size_t min_idx   = 2;
  std::array<int, 3> divided = {1, 1, 1};
  size_t index               = min_idx;
  for (int i = 0; i < shape.size(); ++i) {
    if (index != maj_idx) {
      bool in_dim = std::find(dim.begin(), dim.end(), i) != dim.end();
      if (index == min_idx && in_dim) {
        index = mid_idx;
      } else if (index == mid_idx && !in_dim) {
        index = maj_idx;
      }
    }
    divided[index] *= shape[i];
  }
  return divided;
}

bool ReductionRewritePass::IsHighPerformance(bool on_row, const std::array<int, 3>& dim) const {
  constexpr int warp_size = 32;
  if (on_row) return dim[2] >= warp_size;
  int major_size = dim[1];
  int minor_size = dim[2];
  bool is_emit   = (major_size < warp_size) || (major_size < 2 * warp_size && minor_size < warp_size) ||
                 (major_size < 4 * warp_size && minor_size < warp_size) ||
                 (major_size < 8 * warp_size && minor_size < warp_size);
  return !is_emit;
}

bool ReductionRewritePass::CanRewriteReduction(const Instruction& instr) const {
  const std::set<std::string> types = {"reduce_sum", "reduce_prod"};
  // Limit type of reduction.
  if (types.find(instr->op_type) == types.end()) return false;
  auto shape = instr->inputs[0]->shape;
  auto dim   = instr.GetAttrs<std::vector<int>>("dim");
  std::vector<int> keep_dim;
  for (int i = 0; i < shape.size(); ++i) {
    if (std::find(dim.begin(), dim.end(), i) == dim.end()) {
      keep_dim.push_back(i);
    }
  }
  if (!IsDimConsecutive(dim) && !IsDimConsecutive(keep_dim)) return false;
  if (keep_dim.empty()) {
    int counts = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    return IsHighPerformance(true, {1, 1, counts});
  }
  auto divided = DivideShapeByDim(shape, keep_dim);
  if (divided[1] == 1) return IsHighPerformance(true, {1, 1, divided[0] * divided[2]});
  if (divided[2] == 1) return IsHighPerformance(false, {1, divided[0], divided[1]});
  return IsHighPerformance(false, divided);
}

Instruction ReductionRewritePass::RewriteReduction(const Instruction& instr) const { return instr; }

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(ReductionRewrite) {
  CINN_REGISTER_PROGRAM_PASS(ReductionRewrite, ::cinn::frontend::pass::ReductionRewritePass);

  return true;
}

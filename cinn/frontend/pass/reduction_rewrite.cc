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
  struct ReductionDim {
    bool on_row;
    std::array<int, 3> dim;
  };
  bool IsDimConsecutive(std::vector<int> dim) const;
  std::array<int, 3> DivideShapeByDim(const std::vector<int>& shape, const std::vector<int>& dim) const;
  std::array<int, 3> GetReductionTiling(const ReductionDim& reduction_dim) const;
  ReductionDim GetReductionDim(const std::vector<int>& shape, const std::vector<int>& dim) const;
  bool IsHighPerformance(bool on_row, const std::array<int, 3>& dim) const;
  bool CanRewriteReduction(const Instruction& instr) const;
  std::vector<Instruction> RewriteReduction(const Instruction& instr) const;
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
      for (auto new_instr : RewriteReduction(instr)) {
        builder.AppendInstruction(new_instr);
      }
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

std::vector<Instruction> ReductionRewritePass::RewriteReduction(const Instruction& instr) const {
  auto shape                          = instr->inputs[0]->shape;
  auto dim                            = instr.GetAttrs<std::vector<int>>("dim");
  auto reduction_dim                  = GetReductionDim(shape, dim);
  std::array<int, 3> reduction_tiling = GetReductionTiling(reduction_dim);
  std::vector<int> input_shape_dims   = shape;
  bool reduce_batch_dimension         = dim.size() > 1;
  VLOG(3) << "reduce_batch_dimension = " << reduce_batch_dimension;

  std::sort(dim.begin(), dim.end());
  CHECK_LE(dim.size(), 2);
  int reduced_input_dimension = dim[dim.size() - 1];
  VLOG(3) << "reduced_input_dimension: " << reduced_input_dimension;

  auto RewriteBatchDimensionLargerThanTile =
      [](const Instruction& instr, const ReductionDim& reduction_dim, int reduced_input_dimension) {
        CHECK(reduction_dim.on_row);

        Instruction inner(instr->op_type, instr->inputs);
        inner.SetAttr("dim", std::vector<int>({reduced_input_dimension}));

        Instruction out(instr->op_type, inner->outputs);
        out.SetAttr("dim", std::vector<int>({0}));

        return std::vector<Instruction>({inner, out});
      };
  if (reduce_batch_dimension && shape[0] > 8) {
    return RewriteBatchDimensionLargerThanTile(instr, reduction_dim, reduced_input_dimension);
  }
  bool is_row_reduction = reduction_dim.on_row;

  auto ReductionIsRaceFree = [](const ReductionDim& reduction_dim, const std::array<int, 3>& reduction_tiling) {
    return (reduction_dim.on_row && reduction_dim.dim[2] <= 1024 * reduction_tiling[2] && reduction_dim.dim[0] <= 8) ||
           (!reduction_dim.on_row && reduction_dim.dim[1] <= 32 * reduction_tiling[1]);
  };
  if (ReductionIsRaceFree(reduction_dim, reduction_tiling)) {
    return {instr};
  }

  int reduced_dim_size = shape[reduced_input_dimension];
  VLOG(3) << "reduced_dim_size = " << reduced_dim_size;

  int num_fit = static_cast<int>(std::ceil(std::sqrt(reduced_dim_size)));

  bool no_padding_necessary = reduced_dim_size % num_fit == 0;
  auto padded               = [&]() -> std::array<std::vector<int>, 2> { return {instr->inputs[0]->shape}; }();

  absl::InlinedVector<int, 3> reshaped_dimensions;
  for (int64_t dim_idx = 0; dim_idx < padded[0].size(); dim_idx++) {
    if (dim_idx == reduced_input_dimension) {
      if (no_padding_necessary) {
        reshaped_dimensions.push_back(reduced_dim_size / num_fit);
      } else {
        reshaped_dimensions.push_back(num_fit);
      }

      reshaped_dimensions.push_back(num_fit);
    } else {
      reshaped_dimensions.push_back(padded[0][dim_idx]);
    }
  }

  absl::InlinedVector<int, 3> inner_reduce_dimensions = reshaped_dimensions;
  int inner_reduced_dimension = is_row_reduction ? inner_reduce_dimensions.size() - 1 : reduced_input_dimension;
  VLOG(2) << "inner_reduced_dimension = " << inner_reduced_dimension;
  inner_reduce_dimensions.erase(inner_reduce_dimensions.begin() + inner_reduced_dimension);
  if (reduce_batch_dimension) {
    inner_reduce_dimensions.erase(inner_reduce_dimensions.begin());
  }
  std::vector<int> dims_to_reduce = {inner_reduced_dimension};
  if (reduce_batch_dimension) {
    dims_to_reduce.push_back(0);
    inner_reduced_dimension -= 1;
  }

  std::array<std::vector<int>, 2> reshaped_padded_inputs;
  std::array<std::vector<int>, 2> inner_reduce_shapes;
  for (int i = 0; i < padded.size(); i++) {
    auto& p                         = padded[i];
    std::vector<int> reshaped_shape = {reshaped_dimensions[0], reshaped_dimensions[1], reshaped_dimensions[2]};
  }

  absl::InlinedVector<int, 3> outer_reduce_dimensions = inner_reduce_dimensions;
  int outer_reduced_dimension = is_row_reduction ? outer_reduce_dimensions.size() - 1 : reduced_input_dimension;

  outer_reduce_dimensions.erase(outer_reduce_dimensions.begin() + outer_reduced_dimension);

  return {instr};
}

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(ReductionRewrite) {
  CINN_REGISTER_PROGRAM_PASS(ReductionRewrite, ::cinn::frontend::pass::ReductionRewritePass);

  return true;
}

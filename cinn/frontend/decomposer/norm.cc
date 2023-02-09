// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/frontend/decomposer_registry.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace decomposer {

struct NormHelper {
  NormHelper(NetBuilder* net_builder, const std::vector<int>& arg_x_shape, int32_t axis, std::string bn_op_type) {
    CHECK_GT(arg_x_shape.size(), 0UL) << "The input's shape of norm needs to be greater than 0";

    builder          = net_builder;
    x_shape          = arg_x_shape;
    reduce_dim       = {axis};
    num_instructions = builder->size();
    op_type          = bn_op_type;
  }

  ~NormHelper() {
    VLOG(4) << op_type << " is decomposed to " << builder->size() - num_instructions << " instructions.";
  }

  // square_sum = reduce_sum(x * x)
  Variable SquareSum(Variable x) {
    auto x_square     = builder->Multiply(x, builder->Identity(x));
    auto x_square_sum = Reduce(x_square);

    return x_square_sum;
  }

  // std_square_sum_inv = rsqrt(square_sum + epsilon)
  Variable StdSquareSumInv1d(Variable square_sum, float epsilon) {
    auto epsilon_1d = builder->FillConstant(
        square_sum->shape, epsilon, common::UniqName("norm_epsilon"), common::Type2Str(square_sum->type));
    auto std_square_sum_inv = builder->Rsqrt(builder->Add(square_sum, epsilon_1d));
    return std_square_sum_inv;
  }

  Variable Reduce(Variable x) { return builder->ReduceSum(x, reduce_dim, true); }

  NetBuilder* builder{nullptr};
  std::vector<int> x_shape;
  std::vector<int> reduce_dim;
  std::string op_type;
  int num_instructions{0};
};

void norm(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 1UL) << "The number of the given inputs is not equal to the required for op "
                                      << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL) << "The number of the given outputs is not equal to the required for op "
                                       << instr->op_type;
  auto& x_orig = instr->inputs[0];

  int32_t axis  = instr.GetAttrs<int32_t>("axis");
  float epsilon = instr.GetAttrs<float>("epsilon");

  NetBuilder* builder = context.builder();
  NormHelper helper(builder, x_orig->shape, axis, "norm");

  auto square_sum         = helper.SquareSum(x_orig);
  auto std_square_sum_inv = helper.StdSquareSumInv1d(square_sum, epsilon);
  auto normalized         = builder->Multiply(x_orig, std_square_sum_inv);
  auto y                  = builder->Cast(normalized, common::Type2Str(x_orig->type));

  context.MapOutToOrigin(y, instr->outputs[0]);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(norm_decomposer) {
  CINN_DECOMPOSER_REGISTER(norm, cinn::frontend::decomposer::norm);

  return true;
}

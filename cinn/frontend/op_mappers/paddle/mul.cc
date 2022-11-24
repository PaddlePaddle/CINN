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

#include "absl/types/optional.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void MulOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();

  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);

  // Step1: flatten multi-dimension matrix input to two-dimension matrix
  auto x_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "x_num_col_dims", 1);
  auto y_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "y_num_col_dims", 1);

  auto flatten_shape = [](const cinn::utils::ShapeType& shape, int num_col_dims) {
    if (shape.size() <= 2) {
      return shape;
    }

    if (num_col_dims < 0) {
      num_col_dims += shape.size();
    }

    CHECK_GT(num_col_dims, 0) << "The [num_col_dims] should not be 0 in mul op! Please check.";
    CHECK_LT(num_col_dims, shape.size()) << "The [num_col_dims] > rank(input) in mul op! Please check.";

    cinn::utils::ShapeType new_shape(2, 1);
    for (int i = 0; i < num_col_dims; ++i) {
      new_shape[0] *= shape[i];
    }
    for (int i = num_col_dims; i < shape.size(); ++i) {
      new_shape[1] *= shape[i];
    }
    return new_shape;
  };

  const auto& x_shape = flatten_shape(x->shape, x_num_col_dims);
  const auto& y_shape = flatten_shape(y->shape, y_num_col_dims);

  auto x_reshape = x;
  if (x_shape != x->shape) {
    x_reshape = ctx.Builder()->Reshape(x, x_shape);
  }

  auto y_reshape = y;
  if (y_shape != y->shape) {
    y_reshape = ctx.Builder()->Reshape(y, y_shape);
  }

  // Step2: matmul
  const auto& matmul_out = ctx.Builder()->Matmul(x_reshape, y_reshape);

  // Step3 : recover matmul's output shape
  cinn::utils::ShapeType out_shape;
  for (int i = 0; i < x_num_col_dims; ++i) {
    out_shape.emplace_back(x->shape[i]);
  }
  for (int i = y_num_col_dims; i < y->shape.size(); ++i) {
    out_shape.emplace_back(y->shape[i]);
  }

  auto out = matmul_out;
  if (matmul_out->shape != out_shape) {
    out = ctx.Builder()->Reshape(matmul_out, out_shape);
  }

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_mul) {
  CINN_REGISTER_OP_MAPPER(mul, cinn::frontend::paddle_mappers::MulOpMapper)
  return true;
}

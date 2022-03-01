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

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace science_mappers {

void MatmulOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  VLOG(4) << x_name << " x " << y_name;

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.Builder()->Matmul(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ReduceOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  auto axis    = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "axis");
  auto keepdim = utils::GetAttrOrDefault<bool>(op_desc, "keepdim", false);

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Reudce " << x_name << " from shape (" << cinn::utils::Join(x->shape, ",")
          << "), now only support reduce_sum.";

  // now paddle science only need reduce sum
  auto out = ctx.Builder()->ReduceSum(x, axis, keepdim);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace science_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(science_nn) {
  CINN_REGISTER_OP_MAPPER(matmul_p, cinn::frontend::science_mappers::MatmulOpMapper)
  CINN_REGISTER_OP_MAPPER(reduce_p, cinn::frontend::science_mappers::ReduceOpMapper)
  return true;
}

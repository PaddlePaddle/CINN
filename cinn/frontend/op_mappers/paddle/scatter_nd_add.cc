// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
namespace paddle_mappers {

void ScatterNdAddOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Index").size(), 1UL);
  auto index_name = op_desc.Input("Index").front();
  CHECK_EQ(op_desc.Input("Updates").size(), 1UL);
  auto updates_name = op_desc.Input("Updates").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x       = ctx.GetVar(x_name);
  auto index   = ctx.GetVar(index_name);
  auto updates = ctx.GetVar(updates_name);

  auto axes = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "axes", {});

  // auto shape_name = cinn::utils::TransValidVarName(x_name + "shape");
  // auto shape = ctx.Builder()->Constant(x->shape, shape_name);

  // auto scatter_nd_out = ctx.Builder()->ScatterNd(updates, index, shape, axes);
  auto scatter_nd_out = ctx.Builder()->ScatterNd(updates, index, x->shape);
  VLOG(4) << "scatter_nd_add X:" << x_name << "[" << cinn::utils::Join(x->shape, ",") << "] with index:" << index_name
          << "[" << cinn::utils::Join(index->shape, ",") << "] with updates:" << updates_name << "["
          << cinn::utils::Join(updates->shape, ",") << "] at axes=" << cinn::utils::Join(axes, ",");

  auto out = ctx.Builder()->Add(x, scatter_nd_out);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_scatter_nd_add) {
  CINN_REGISTER_OP_MAPPER(scatter_nd_add, cinn::frontend::paddle_mappers::ScatterNdAddOpMapper)
  return true;
}

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

void ConcatOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("X").size(), 1UL);
  auto x_names = op_desc.Input("X");
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  Variable out;
  if (x_names.size() == 1) {
    // if concat only has one input, using Identity to copy the input and return
    auto x = ctx.GetVar(x_names.front());
    out    = ctx.Builder()->Identity(x);
  } else {
    std::vector<Variable> xs;
    for (const auto& name : x_names) {
      xs.emplace_back(ctx.GetVar(name));
    }

    auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis");

    out = ctx.Builder()->Concat(xs, axis);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ReshapeOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();

  auto shape = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "shape");

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Reshape " << x_name << "from shape (" << cinn::utils::Join(x->shape, ",") << ") to ("
          << cinn::utils::Join(shape, ",") << ").";

  auto out = ctx.Builder()->Reshape(x, shape);

  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void TransposeOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  auto x = ctx.GetVar(x_name);

  CHECK(x->shape.size() == 2) << "Now transpose_p only support 2-dim matrix.";
  VLOG(4) << "Transpose " << x_name << " with shape (" << cinn::utils::Join(x->shape, ",") << ").";

  auto out = ctx.Builder()->Transpose(x, {1, 0});

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SliceSelectOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  CHECK(op_desc.HasAttr("starts"));
  auto starts = op_desc.GetAttr<std::vector<int>>("starts");
  CHECK(op_desc.HasAttr("ends"));
  auto ends = op_desc.GetAttr<std::vector<int>>("ends");
  CHECK(op_desc.HasAttr("axis"));
  auto axis = op_desc.GetAttr<std::vector<int>>("axis");
  CHECK(op_desc.HasAttr("strides"));
  auto strides = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides");

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Slice " << x_name << "from shape (" << cinn::utils::Join(x->shape, ",") << ") with starts ["
          << cinn::utils::Join(starts, ",") << "], ends [" << cinn::utils::Join(ends, ",") << "], axis ["
          << cinn::utils::Join(axis, ",") << "], strides [" << cinn::utils::Join(strides, ",") << "].";

  auto out = ctx.Builder()->Slice(x, axis, starts, ends, std::vector<int>{}, strides);

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

CINN_REGISTER_HELPER(science_transform) {
  CINN_REGISTER_OP_MAPPER(concat_p, cinn::frontend::science_mappers::ConcatOpMapper)
  CINN_REGISTER_OP_MAPPER(reshape_p, cinn::frontend::science_mappers::ReshapeOpMapper)
  CINN_REGISTER_OP_MAPPER(transpose_p, cinn::frontend::science_mappers::TransposeOpMapper)
  CINN_REGISTER_OP_MAPPER(slice_select_p, cinn::frontend::science_mappers::SliceSelectOpMapper)
  CINN_REGISTER_OP_MAPPER(reduce_p, cinn::frontend::science_mappers::ReduceOpMapper)
  return true;
}

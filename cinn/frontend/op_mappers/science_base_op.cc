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
namespace op_mappers {

void FillConstantPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto y_name = op_desc.Output("Y").front();

  auto shape = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "shape");
  // TODO(jiangcheng): value support different datatype, not just float
  auto value = utils::GetAttrOrDefault<float>(op_desc, "value");

  VLOG(4) << "fill constant (" << value << ") with shape (" << cinn::utils::Join(shape, ",") << ").";

  const auto& cinn_name = cinn::utils::TransValidVarName(y_name);
  CheckVarNameValid(cinn_name);

  auto out = ctx.Builder()->FillConstant<float>(shape, value, cinn_name);

  ctx.AddVar(y_name, out);
  ctx.AddVarModelToProgram(y_name, out->id);
}

void BroadcastPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto y_name = op_desc.Output("Y").front();

  std::vector<int> y_shape;
  if (op_desc.HasInput("ShapeTensor")) {
    CHECK_EQ(op_desc.Input("ShapeTensor").size(), 1UL);
    auto shape_name = op_desc.Input("ShapeTensor").front();

    auto shape_var = ctx.GetVar(x_name);
    // Can we get the variable's real data at OpMapper?
    // y_shape = GetVarData<std::vector<int>>(shape_var);
  } else if (op_desc.HasAttr("shape")) {
    y_shape = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "shape");
  } else {
    LOG(FATAL) << "The broadcast_p operator should has 'shape' parameter, but " << x_name << "'s broadcast hasn't.";
  }

  auto x = ctx.GetVar(x_name);

  auto x_shape_size = x->shape.size();
  auto y_shape_size = y_shape.size();
  CHECK(x_shape_size == y_shape_size) << "The broadcast_p operator's input "
                                      << "shape size should the same as output "
                                      << " shape size, but here (" << x_shape_size << " vs " << y_shape_size << ").";

  std::vector<int> broadcast_axes(x_shape_size, 0);
  for (int i = 0; i < x_shape_size; ++i) {
    broadcast_axes[i] = i;
  }

  VLOG(4) << "Broadcast " << x_name << "from shape (" << cinn::utils::Join(x->shape, ",") << ") to shape ("
          << cinn::utils::Join(y_shape, ",") << ").";

  auto out = ctx.Builder()->BroadcastTo(x, y_shape, broadcast_axes);

  ctx.AddVar(y_name, out);
  ctx.AddVarModelToProgram(y_name, out->id);
}

void AddPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  VLOG(4) << x_name << " + " << y_name;

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.Builder()->Add(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SubPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  VLOG(4) << x_name << " - " << y_name;

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.Builder()->Sub(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void DivPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  VLOG(4) << x_name << " / " << y_name;

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.Builder()->Div(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void MulPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  VLOG(4) << x_name << " .* " << y_name;

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.Builder()->ElementwiseMul(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SqrtPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Compute " << x_name << " 's sqrt result with shape (" << cinn::utils::Join(x->shape, ",") << ").";

  // now paddle science only need reduce sum
  auto out = ctx.Builder()->Sqrt(x);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void TanhPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Compute " << x_name << " 's tanh result with shape (" << cinn::utils::Join(x->shape, ",") << ").";

  // now paddle science only need reduce sum
  auto out = ctx.Builder()->Tanh(x);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void MatmulPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void ReducePOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void ConcatPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("X").size(), 1UL);
  auto x_names = op_desc.Input("X");
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto out_name = op_desc.Output("Y").front();

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    xs.emplace_back(ctx.GetVar(name));
  }

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis");

  auto out = ctx.Builder()->Concat(xs, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ReshapePOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void TransposePOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void SliceSelectPOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(science) {
  CINN_REGISTER_OP_MAPPER(fill_constant_p, cinn::frontend::op_mappers::FillConstantPOpMapper)
  CINN_REGISTER_OP_MAPPER(broadcast_p, cinn::frontend::op_mappers::BroadcastPOpMapper)

  CINN_REGISTER_OP_MAPPER(add_p, cinn::frontend::op_mappers::AddPOpMapper)
  CINN_REGISTER_OP_MAPPER(sub_p, cinn::frontend::op_mappers::SubPOpMapper)
  CINN_REGISTER_OP_MAPPER(div_p, cinn::frontend::op_mappers::DivPOpMapper)
  CINN_REGISTER_OP_MAPPER(mul_p, cinn::frontend::op_mappers::MulPOpMapper)

  CINN_REGISTER_OP_MAPPER(sqrt_p, cinn::frontend::op_mappers::SqrtPOpMapper)
  CINN_REGISTER_OP_MAPPER(tanh_p, cinn::frontend::op_mappers::TanhPOpMapper)
  CINN_REGISTER_OP_MAPPER(matmul_p, cinn::frontend::op_mappers::MatmulPOpMapper)
  CINN_REGISTER_OP_MAPPER(reduce_p, cinn::frontend::op_mappers::ReducePOpMapper)

  CINN_REGISTER_OP_MAPPER(concat_p, cinn::frontend::op_mappers::ConcatPOpMapper)
  CINN_REGISTER_OP_MAPPER(reshape_p, cinn::frontend::op_mappers::ReshapePOpMapper)
  CINN_REGISTER_OP_MAPPER(transpose_p, cinn::frontend::op_mappers::TransposePOpMapper)

  CINN_REGISTER_OP_MAPPER(slice_select_p, cinn::frontend::op_mappers::SliceSelectPOpMapper)
  return true;
}

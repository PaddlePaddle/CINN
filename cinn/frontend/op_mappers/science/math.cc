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

void AddOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void SubOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void DivOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void MulOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Z").size(), 1UL);
  auto out_name = op_desc.Output("Z").front();

  VLOG(4) << x_name << " .* " << y_name;

  Variable out;
  if (x_name != y_name) {
    auto x = ctx.GetVar(x_name);
    auto y = ctx.GetVar(y_name);
    out    = ctx.Builder()->ElementwiseMul(x, y);
  } else {
    auto x = ctx.GetVar(x_name);
    auto y = ctx.Builder()->Identity(x);
    out    = ctx.Builder()->ElementwiseMul(x, y);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SqrtOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void TanhOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

void MatMulOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
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

}  // namespace science_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(science_math) {
  CINN_REGISTER_OP_MAPPER(add_p, cinn::frontend::science_mappers::AddOpMapper)
  CINN_REGISTER_OP_MAPPER(sub_p, cinn::frontend::science_mappers::SubOpMapper)
  CINN_REGISTER_OP_MAPPER(div_p, cinn::frontend::science_mappers::DivOpMapper)
  CINN_REGISTER_OP_MAPPER(mul_p, cinn::frontend::science_mappers::MulOpMapper)
  CINN_REGISTER_OP_MAPPER(sqrt_p, cinn::frontend::science_mappers::SqrtOpMapper)
  CINN_REGISTER_OP_MAPPER(tanh_p, cinn::frontend::science_mappers::TanhOpMapper)
  CINN_REGISTER_OP_MAPPER(matmul_p, cinn::frontend::science_mappers::MatMulOpMapper)
  return true;
}

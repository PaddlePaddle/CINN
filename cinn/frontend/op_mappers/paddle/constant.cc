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

#include <algorithm>

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"
#include "cinn/frontend/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ShapeOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x   = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Constant(x->shape, cinn::utils::TransValidVarName(out_name));

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void FillConstantOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto y_name = op_desc.Output("Out").front();

  auto shape     = utils::ToShapeType(utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "shape"));
  auto value     = utils::GetAttrOrDefault<float>(op_desc, "value", 0.0f);
  auto force_cpu = utils::GetAttrOrDefault<bool>(op_desc, "force_cpu", false);

  auto dtype_id = utils::GetAttrOrDefault<int>(op_desc, "dtype", static_cast<int>(paddle::cpp::VarDescAPI::Type::FP32));
  auto dtype_pd = static_cast<paddle::cpp::VarDescAPI::Type>(dtype_id);
  auto dtype_cinn = utils::CppVarType2CommonType(dtype_pd);
  auto dtype      = common::Type2Str(dtype_cinn);

  VLOG(4) << "fill constant (" << value << ") with shape (" << cinn::utils::Join(shape, ",") << ") and dtype [" << dtype
          << "]";

  const auto& cinn_name = cinn::utils::TransValidVarName(y_name);
  CheckVarNameValid(cinn_name);

  auto out = ctx.Builder()->FillConstant(shape, value, cinn_name, dtype, force_cpu);

  ctx.AddVar(y_name, out);
  ctx.AddVarModelToProgram(y_name, out->id);
}

void FillAnyLikeOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  auto x      = ctx.GetVar(x_name);

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto y_name = op_desc.Output("Out").front();

  auto shape = utils::ToShapeType(x->shape);
  auto value = utils::GetAttrOrDefault<float>(op_desc, "value");

  auto dtype_id = utils::GetAttrOrDefault<int>(op_desc, "dtype", static_cast<int>(paddle::cpp::VarDescAPI::Type::FP32));
  cinn::common::Type dtype_cinn;
  if (dtype_id < 0) {
    dtype_cinn = x->type;
  } else {
    auto dtype_pd = static_cast<paddle::cpp::VarDescAPI::Type>(dtype_id);
    dtype_cinn    = utils::CppVarType2CommonType(dtype_pd);
  }

  auto dtype = common::Type2Str(dtype_cinn);

  VLOG(4) << "FillAnyLikeOp: fill constant (" << value << ") with shape (" << cinn::utils::Join(shape, ", ")
          << ") and dtype [" << dtype << "]";

  const auto& cinn_name = cinn::utils::TransValidVarName(y_name);
  CheckVarNameValid(cinn_name);

  auto out = ctx.Builder()->FillConstant(shape, value, cinn_name, dtype);

  ctx.AddVar(y_name, out);
  ctx.AddVarModelToProgram(y_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_constant) {
  CINN_REGISTER_OP_MAPPER(shape, cinn::frontend::paddle_mappers::ShapeOpMapper)
  CINN_REGISTER_OP_MAPPER(fill_constant, cinn::frontend::paddle_mappers::FillConstantOpMapper)
  CINN_REGISTER_OP_MAPPER(fill_any_like, cinn::frontend::paddle_mappers::FillAnyLikeOpMapper)
  return true;
}

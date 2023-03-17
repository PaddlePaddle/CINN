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

#include <iostream>

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"
#include "cinn/frontend/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ArgMaxOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x        = ctx.GetVar(x_name);
  auto axis     = op_desc.GetAttr<int32_t>("axis");
  auto keepdims = op_desc.GetAttr<bool>("keepdims");
  auto flatten  = op_desc.GetAttr<bool>("flatten");
  auto dtype_id =
      utils::GetAttrOrDefault<int>(op_desc, "dtype", static_cast<int>(paddle::cpp::VarDescAPI::Type::INT64));
  if (dtype_id < 0) dtype_id = static_cast<int>(paddle::cpp::VarDescAPI::Type::INT64);
  auto dtype_pd   = static_cast<paddle::cpp::VarDescAPI::Type>(dtype_id);
  auto dtype_cinn = utils::CppVarType2CommonType(dtype_pd);
  auto dtype      = common::Type2Str(dtype_cinn);

  std::cout << "get arg max all attr" << std::endl;
  std::cout << axis << ' ' << keepdims << ' ' << flatten << ' ' << dtype << std::endl;
  int ndim = x->shape.size();
  // If flatten = true, flatten x and do argmax on axis 0.
  if (flatten) {
    x    = ctx.Builder()->Reshape(x, {-1});
    axis = 0;
    ndim = x->shape.size();
  }
  std::cout << "after flatten all attr" << std::endl;
  auto out = ctx.Builder()->Argmax(x, axis, keepdims);
  std::cout << "after build.argmax" << std::endl;
  std::cout << "out:" << std::endl;
  std::cout << out << std::endl;
  out = ctx.Builder()->Cast(out, dtype);
  std::cout << "after build.cast" << std::endl;
  ctx.AddVar(out_name, out);
  std::cout << "after add var" << std::endl;
  ctx.AddVarModelToProgram(out_name, out->id);
  std::cout << "argmax end" << std::endl;
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_arg_max) {
  CINN_REGISTER_OP_MAPPER(arg_max, cinn::frontend::paddle_mappers::ArgMaxOpMapper)
  return true;
}
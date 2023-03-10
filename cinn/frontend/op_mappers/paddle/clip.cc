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
#include "glog/logging.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ClipOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  auto x        = ctx.GetVar(x_name);
  auto builder  = ctx.Builder();

  if (op_desc.HasInput("Min") && op_desc.Input("Min").size() > 0) {
    CHECK_EQ(op_desc.Input("Min").size(), 1) << "clip op should have only one input for Min";
    auto min_val_name   = op_desc.Input("Min").front();
    auto min_val_tensor = ctx.GetVar(min_val_name);
    CHECK_EQ(x->type, min_val_tensor->type_info()) << "The input X and Min should have the same type";
    CHECK_EQ(min_val_tensor->shape.size(), 1UL);
    min_val_tensor = builder->BroadcastTo(min_val_tensor, x->shape);
    x              = builder->Max(x, min_val_tensor);
  } else if (op_desc.HasAttr("min")) {
    auto min_value = op_desc.GetAttr<float>("min");
    auto min_val_tensor =
        builder->FillConstant(x->shape, min_value, common::UniqName("constant"), common::Type2Str(x->type));
    x = builder->Max(x, min_val_tensor);
  }

  if (op_desc.HasInput("Max") && op_desc.Input("Max").size() > 0) {
    CHECK_EQ(op_desc.Input("Max").size(), 1) << "clip op should have only one input for Max";
    auto max_val_name   = op_desc.Input("Max").front();
    auto max_val_tensor = ctx.GetVar(max_val_name);
    CHECK_EQ(x->type, max_val_tensor->type_info()) << "The input X and Max should have the same type";
    CHECK_EQ(max_val_tensor->shape.size(), 1UL);
    max_val_tensor = builder->BroadcastTo(max_val_tensor, x->shape);
    x              = builder->Min(x, max_val_tensor);
  } else if (op_desc.HasAttr("max")) {
    auto max_value = op_desc.GetAttr<float>("max");
    auto max_val_tensor =
        builder->FillConstant(x->shape, max_value, common::UniqName("constant"), common::Type2Str(x->type));
    x = builder->Min(x, max_val_tensor);
  }

  ctx.AddVar(out_name, x);
  ctx.AddVarModelToProgram(out_name, x->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_clip) {
  CINN_REGISTER_OP_MAPPER(clip, cinn::frontend::paddle_mappers::ClipOpMapper)
  return true;
}

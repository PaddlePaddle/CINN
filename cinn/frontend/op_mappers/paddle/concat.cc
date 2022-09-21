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

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ConcatOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("X").size(), 1UL);
  auto x_names = op_desc.Input("X");
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", 0);

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    xs.emplace_back(ctx.GetVar(name));
  }

  auto err_x = std::find_if(xs.begin(), xs.end(), [&](Variable x) { return x->type != xs.front()->type; });
  CHECK(err_x == xs.end()) << "All input's dtype of [concat] should be the same, be the input " << (*err_x)->id
                           << "'s dtype [" << (*err_x)->type << "] not equal to the first input " << xs.front()->id
                           << "'s dtype [" << xs.front()->type << "]";

  Variable out;
  if (xs.size() == 1) {
    // if concat only has one input, using Identity to copy the input and return
    out = ctx.Builder()->Identity(xs.front());
  } else {
    out = ctx.Builder()->Concat(xs, axis);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void StackOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("X").size(), 1UL);
  auto x_names = op_desc.Input("X");

  std::string out_name;
  if (op_desc.HasOutput("Out")) {
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    out_name = op_desc.Output("Out").front();
  } else if (op_desc.HasOutput("Y")) {
    CHECK_EQ(op_desc.Output("Y").size(), 1UL);
    out_name = op_desc.Output("Y").front();
  } else {
    LOG(FATAL) << "The output argument name of [stack] should be 'Out' or 'Y', but here cannot found! Please check.";
  }

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", 0);

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    xs.emplace_back(ctx.GetVar(name));
  }

  auto err_x = std::find_if(xs.begin(), xs.end(), [&](Variable x) { return x->type != xs.front()->type; });
  CHECK(err_x == xs.end()) << "All input's dtype of [concat] should be the same, be the input " << (*err_x)->id
                           << "'s dtype [" << (*err_x)->type << "] not equal to the first input " << xs.front()->id
                           << "'s dtype [" << xs.front()->type << "]";

  err_x = std::find_if(xs.begin(), xs.end(), [&](Variable x) { return x->shape != xs.front()->shape; });
  CHECK(err_x == xs.end()) << "All input shape of [stack] should be the same, be the input " << (*err_x)->id
                           << "'s shape [" << cinn::utils::Join((*err_x)->shape, ", ") << "] not equal to "
                           << "the first input " << xs.front()->id << "'s shape ["
                           << cinn::utils::Join(xs.front()->shape, ", ") << "]";

  Variable out;
  if (xs.size() == 1) {
    // if stack only has one input, using Identity to copy the input and return
    out = ctx.Builder()->Identity(xs.front());
  } else {
    out = ctx.Builder()->Concat(xs, axis);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_concat) {
  CINN_REGISTER_OP_MAPPER(concat, cinn::frontend::paddle_mappers::ConcatOpMapper)
  CINN_REGISTER_OP_MAPPER(stack, cinn::frontend::paddle_mappers::StackOpMapper)
  return true;
}

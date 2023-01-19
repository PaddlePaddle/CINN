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

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ReduceOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx, const std::string& reduce_type) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis       = utils::ToShapeType(utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dim"));
  auto keepdim    = utils::GetAttrOrDefault<bool>(op_desc, "keep_dim", false);
  auto reduce_all = utils::GetAttrOrDefault<bool>(op_desc, "reduce_all", false);

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Reudce " << reduce_type << " x:" << x_name << " from shape (" << cinn::utils::Join(x->shape, ",")
          << "), with dim=[" << cinn::utils::Join(axis, ",") << "], keepdim=" << keepdim
          << ", reduce_all=" << reduce_all;

  if (reduce_all) {
    axis.clear();
  }

  // now paddle science only need reduce sum
  absl::optional<Variable> out;
  if (reduce_type == "Sum") {
    out = ctx.Builder()->ReduceSum(x, axis, keepdim);
  } else if (reduce_type == "Prod") {
    out = ctx.Builder()->ReduceProd(x, axis, keepdim);
  } else if (reduce_type == "Max") {
    out = ctx.Builder()->ReduceMax(x, axis, keepdim);
  } else if (reduce_type == "Min") {
    out = ctx.Builder()->ReduceMin(x, axis, keepdim);
  } else if (reduce_type == "All") {
    out = ctx.Builder()->ReduceAll(x, axis, keepdim);
  } else if (reduce_type == "Any") {
    out = ctx.Builder()->ReduceAny(x, axis, keepdim);
  }

  CHECK(out) << "Not support Reduce " << reduce_type << "! Please check.";

  ctx.AddVar(out_name, out.value());
  ctx.AddVarModelToProgram(out_name, out.value()->id);
}

#define EXPAND_REDUCE_OPMAPPER(ReduceType)                                                            \
  void Reduce##ReduceType##OpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) { \
    ReduceOpMapper(op_desc, ctx, #ReduceType);                                                        \
  }

EXPAND_REDUCE_OPMAPPER(Sum)
EXPAND_REDUCE_OPMAPPER(Prod)
EXPAND_REDUCE_OPMAPPER(Max)
EXPAND_REDUCE_OPMAPPER(Min)
EXPAND_REDUCE_OPMAPPER(All)
EXPAND_REDUCE_OPMAPPER(Any)
#undef EXPAND_REDUCE_OPMAPPER

void ReduceMeanOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis       = utils::ToShapeType(utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dim"));
  auto keepdim    = utils::GetAttrOrDefault<bool>(op_desc, "keep_dim", false);
  auto reduce_all = utils::GetAttrOrDefault<bool>(op_desc, "reduce_all", false);

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "ReudceMean x:" << x_name << " from shape (" << cinn::utils::Join(x->shape, ",") << "), with dim=["
          << cinn::utils::Join(axis, ",") << "], keepdim=" << keepdim << ", reduce_all=" << reduce_all;

  if (reduce_all) {
    axis.clear();
  }

  int num = 1;
  if (axis.empty()) {
    num = std::accumulate(x->shape.begin(), x->shape.end(), 1, std::multiplies<int>());
  } else {
    for (int i = 0; i < axis.size(); ++i) {
      num *= x->shape[axis[i]];
    }
  }

  auto x_shape = x->shape;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
      x_shape[i] = 1;
    }
  }

  const auto& sum  = ctx.Builder()->ReduceSum(x, axis, false);
  const auto& size = ctx.Builder()->FillConstant(
      sum->shape, num, cinn::common::UniqName(x->id + "_mean"), cinn::common::Type2Str(sum->type));
  const auto& m_out = ctx.Builder()->Divide(sum, size);
  const auto& out   = ctx.Builder()->Reshape(m_out, x_shape);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_reduce) {
#define EXPAND_REDUCE_OP_MAPPER_REGISTER(op_name, ReduceType) \
  CINN_REGISTER_OP_MAPPER(op_name, cinn::frontend::paddle_mappers::Reduce##ReduceType##OpMapper)

  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_sum, Sum)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_prod, Prod)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_max, Max)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_min, Min)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_all, All)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_any, Any)
#undef EXPAND_REDUCE_OP_MAPPER_REGISTER

  CINN_REGISTER_OP_MAPPER(reduce_mean, cinn::frontend::paddle_mappers::ReduceMeanOpMapper)
  return true;
}

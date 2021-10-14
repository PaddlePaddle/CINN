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

#include "cinn/common/macros.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"
#include "cinn/frontend/op_mappers/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void FetchOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto output_name = op_desc.Input("X").front();
  LOG(INFO) << "detect model output: [" << output_name << "]";
}

void FeedOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  auto outs = op_desc.Output("Out");
  CHECK_EQ(outs.size(), 1UL);
  VLOG(2) << "Model get feed [" << outs[0] << "]";

  auto type = common::Float(32);
  std::vector<int> shape;

  auto var_desc = ctx.GetVarDesc(outs[0]);
  if (var_desc != nullptr) {
    type = utils::CppVarType2CommonType(var_desc->GetDataType());
    for (auto num : var_desc->GetShape()) {
      shape.emplace_back(static_cast<int>(num));
    }
  } else {
    LOG(WARNING) << "Feed VarDesc [" << outs[0] << "] not found";
  }

  Placeholder input(type, shape, outs[0]);
  ctx.AddVar(outs[0], input);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(fetch_feed) {
  CINN_REGISTER_OP_MAPPER(fetch, cinn::frontend::op_mappers::FetchOpMapper)
  CINN_REGISTER_OP_MAPPER(feed, cinn::frontend::op_mappers::FeedOpMapper)
  return true;
}

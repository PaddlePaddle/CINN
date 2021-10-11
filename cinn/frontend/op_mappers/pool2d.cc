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

void Pool2dOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("pooling_type"));
  auto pooling_type = op_desc.GetAttr<std::string>("pooling_type");
  CHECK(op_desc.HasAttr("ksize"));
  auto ksize = op_desc.GetAttr<std::vector<int>>("ksize");

  auto strides      = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto padding_size = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});

  if (padding_size.size() == 2) {
    padding_size.insert(padding_size.begin(), padding_size.front());
    padding_size.push_back(padding_size.back());
  }

  auto ceil_mode         = utils::GetAttrOrDefault<bool>(op_desc, "ceil_mode", false);
  auto exclusive         = utils::GetAttrOrDefault<bool>(op_desc, "exclusive", true);
  auto global_pooling    = utils::GetAttrOrDefault<bool>(op_desc, "global_pooling", false);
  auto data_format       = utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "NCHW");
  auto adaptive          = utils::GetAttrOrDefault<bool>(op_desc, "adaptive", false);
  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(op_desc, "padding_algorithm", "EXPLICIT");
  auto x                 = ctx.GetVar(x_name);
  auto out               = ctx.builder_->pool2d(x,
                                  pooling_type,
                                  ksize,
                                  strides,
                                  padding_size,
                                  ceil_mode,
                                  exclusive,
                                  global_pooling,
                                  data_format,
                                  adaptive,
                                  padding_algorithm);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(pool2d) {
  CINN_REGISTER_OP_MAPPER(pool2d, cinn::frontend::op_mappers::Pool2dOpMapper)
  return true;
}

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

#include "cinn/frontend/paddle_model_to_netbuilder.h"

#include <glog/logging.h>

#include <utility>

#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/frontend/paddle/cpp/program_desc.h"
#include "cinn/frontend/paddle/model_parser.h"
#include "cinn/hlir/op/use_ops.h"

namespace cinn {
namespace frontend {

void PaddleModelToNetBuilder::RunOp(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  const auto& op_type = op_desc.Type();
  auto kernel         = OpMapperRegistry::Global()->Find(op_type);
  CHECK(kernel) << "Not supported op [" << op_type << "] found";
  kernel->Run(op_desc, ctx);
}

std::unique_ptr<NetBuilder> PaddleModelToNetBuilder::operator()(const std::string& model_dir, bool is_combined) {
  paddle::cpp::ProgramDesc program_desc;
  paddle::LoadModelPb(model_dir, "__model__", "", scope_, &program_desc, is_combined, false, target_);
  CHECK_EQ(program_desc.BlocksSize(), 1) << "CINN can only support the model with a single block";
  auto* block_desc = program_desc.GetBlock<paddle::cpp::BlockDesc>(0);

  // unique builder name like program_1_of_12
  std::string builder_name = "program_";
  if (program_desc.HasVersion()) {
    builder_name.append(std::to_string(program_desc.Version()));
  }
  builder_name.append("_of_");
  static uint64_t unique_invoke_number = 0;
  builder_name.append(std::to_string(unique_invoke_number++));

  std::unique_ptr<NetBuilder> builder = std::make_unique<NetBuilder>(builder_name);
  OpMapperContext ctx(scope_, target_, builder.get(), &var_map_, &var_model_to_program_map_);

  for (int i = 0; i < block_desc->OpsSize(); i++) {
    auto* op_desc = block_desc->GetOp<paddle::cpp::OpDesc>(i);
    RunOp(*op_desc, ctx);
  }
  return std::move(builder);
}

}  // namespace frontend
}  // namespace cinn

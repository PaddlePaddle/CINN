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

#include "cinn/frontend/paddle_model_convertor.h"

#include <glog/logging.h>

#include <utility>

#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/frontend/paddle/cpp/program_desc.h"
#include "cinn/frontend/paddle/model_parser.h"
#include "cinn/frontend/var_type_utils.h"
#include "cinn/hlir/op/use_ops.h"

namespace cinn {
namespace frontend {
namespace utils {
struct VarInfo {
  std::vector<int> shape;
  common::Type type;
};

VarInfo GetVarInfoFromDesc(paddle::cpp::VarDesc* desc) {
  VarInfo info;
  for (auto num : desc->GetShape()) {
    info.shape.emplace_back(static_cast<int>(num));
  }
  info.type = CppVarType2CommonType(desc->GetDataType());
  return info;
}
}  // namespace utils

void PaddleModelConvertor::PrepareRun(paddle::cpp::BlockDesc* block_desc, const OpMapperContext& ctx) {
  // preserve var desc info lik shape and dtype
  absl::flat_hash_map<std::string, utils::VarInfo> var_info_map;
  for (int i = 9; i < block_desc->VarsSize(); i++) {
    auto* var_desc                 = block_desc->GetVar<paddle::cpp::VarDesc>(i);
    var_info_map[var_desc->Name()] = utils::GetVarInfoFromDesc(var_desc);
  }

  for (int i = 0; i < block_desc->OpsSize(); i++) {
    auto* op_desc       = block_desc->GetOp<paddle::cpp::OpDesc>(i);
    const auto& op_type = op_desc->Type();

    if (op_type == "feed") {
      // if the op is feed op, create output variable here instead of in feed mapper
      // so that we can pass the shape and dtype info into CINN
      const auto& var_names = op_desc->output_vars();
      for (const auto& var_name : var_names) {
        if (var_info_map.count(var_name)) {
          const auto& var = var_info_map.at(var_name);
          auto input      = ctx.Builder()->CreateInput(var.type, var.shape, var_name);
          ctx.AddVar(var_name, input);
        } else {
          LOG(WARNING) << "Var [" << var_name << "] not found in block, using default value";
          auto input = ctx.Builder()->CreateInput(common::Float(32), {}, var_name);
          ctx.AddVar(var_name, input);
        }
        VLOG(4) << "Add feed variable [" << var_name << "]";
      }
    }
  }
}

void PaddleModelConvertor::RunOp(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  const auto& op_type = op_desc.Type();
  auto kernel         = OpMapperRegistry::Global()->Find(op_type);
  CHECK(kernel) << "Not supported op [" << op_type << "] found";
  VLOG(4) << "Running Op " << op_type;
  kernel->Run(op_desc, ctx);
}

Program PaddleModelConvertor::operator()(const std::string& model_dir, bool is_combined) {
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
  VLOG(4) << "NetBuilder Name " << builder_name;

  NetBuilder builder(builder_name);
  OpMapperContext ctx(scope_, target_, &builder, &var_map_, &var_model_to_program_map_);

  PrepareRun(block_desc, ctx);
  for (int i = 0; i < block_desc->OpsSize(); i++) {
    auto* op_desc = block_desc->GetOp<paddle::cpp::OpDesc>(i);
    RunOp(*op_desc, ctx);
  }
  return builder.Build();
}

}  // namespace frontend
}  // namespace cinn

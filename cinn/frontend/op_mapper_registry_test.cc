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

#include <gtest/gtest.h>

#include <memory>
#include <typeinfo>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/utils/registry.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace frontend {

using ::cinn::common::Target;
using ::cinn::frontend::paddle::cpp::OpDesc;
using ::cinn::hlir::framework::Scope;
using ::cinn::hlir::framework::Tensor;
using ::cinn::utils::TransValidVarName;

TEST(OpMapperRegistryTest, basic) {
  auto kernel = OpMapperRegistry::Global()->Find("sigmoid");
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(typeid(*kernel), typeid(OpMapper));
  ASSERT_EQ(kernel->name, "sigmoid");
}

// Test that the reverse of HW gets same result
// between OpMapper and PaddleModelToProgram
TEST(OpMapperRegistryTest, conv2d_reverse) {
  std::unique_ptr<OpDesc> op_desc;
  op_desc->SetType("conv2d");
  op_desc->SetInput("Input", {"input"});
  op_desc->SetInput("Filter", {"filter"});
  op_desc->SetOutput("Output", {"output"});

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::shared_ptr<Scope> net_scope = Scope::Create();
  NetBuilder net_builder("net_builder_name");

  net_scope->Var<Tensor>("input");
  net_scope->Var<Tensor>("filter");

  std::unordered_map<std::string, Variable> var_map;
  std::unordered_map<std::string, std::string> var_model_to_program_map({{"input", "input"}, {"filter", "filter"}});

  OpMapperContext op_mapper_ctx(*net_scope, target, &net_builder, &var_map, &var_model_to_program_map);

  const OpMapper* op_mapper = OpMapperRegistry::Global()->Find("conv2d");
  op_mapper->Run(*op_desc, op_mapper_ctx);
}

}  // namespace frontend
}  // namespace cinn

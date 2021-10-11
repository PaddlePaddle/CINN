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
#include <gtest/gtest.h>

#include "cinn/runtime/use_extern_funcs.h"

DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

TEST(PaddleModelToNetBuilder, basic) {
  auto scope  = hlir::framework::Scope::Create();
  auto target = common::DefaultHostTarget();

  PaddleModelToNetBuilder model_transform(scope.get(), target);
  auto builder = model_transform(FLAGS_model_dir);

  const auto& var_map                  = model_transform.var_map();
  const auto& var_model_to_program_map = model_transform.var_model_to_program_map();

  ASSERT_FALSE(var_map.empty());
  ASSERT_FALSE(var_model_to_program_map.empty());
  LOG(INFO) << builder->name();
}

}  // namespace frontend
}  // namespace cinn

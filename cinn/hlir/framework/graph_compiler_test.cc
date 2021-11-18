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

#include "cinn/hlir/framework/graph_compiler.h"

#include <gtest/gtest.h>

#include "cinn/frontend/net_builder.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace hlir {
namespace framework {

using common::Float;
using frontend::Placeholder;

TEST(GraphCompilerTest, TestRemoveInvaildVariables) {
  frontend::NetBuilder builder("test");
  auto a = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
  auto b = builder.CreateInput(Float(32), {64}, "B");

  auto c = builder.elementwise_add(a, b, 1);
  auto d = builder.relu(c);

  auto target = common::DefaultHostTarget();
  auto graph  = std::make_shared<Graph>(builder.Build(), target);

  // OpFusion will fuse add+relu, and the intermediate variable 'c' is eliminated
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  ASSERT_EQ(scope->var_names().size(), 4);
  EXPECT_NE(scope->FindVar(c->id), nullptr);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  ASSERT_EQ(scope->var_names().size(), 3);
  EXPECT_EQ(scope->FindVar(c->id), nullptr);

  ASSERT_NO_THROW(runtime_program->Execute());
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

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

// Copyright (c) 202 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include <memory>

#include "cinn/cinn.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/utils/data_util.h"

DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

using hlir::framework::Scope;
using utils::Join;

TEST(common_subexpression_elimination, common_subexpression_elimination_case1) {
  Placeholder A(Float(32), {32, 16}, "A");
  Placeholder B(Float(32), {32, 1}, "B", true);

  Program program;
  auto add_1 = program.add(A, B);
  auto add_2 = program.add(A, B);
  auto add   = program.add(add_1, add_2);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "CommonSubexpressionEliminationPass");
  auto scope = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  auto& prerun_instrs  = runtime_program->GetPreRunInstructions();
  auto& run_instrs     = runtime_program->GetRunInstructions();
  ASSERT_EQ(prerun_instrs.size(), 0);
  ASSERT_EQ(run_instrs.size(), 2);

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);

  runtime_program->Execute();
}

TEST(common_subexpression_elimination, common_subexpression_elimination_case2) {
  Placeholder A(Float(32), {32, 16}, "A");
  Placeholder B(Float(32), {32, 1}, "B", true);

  Program program;
  auto sub_1 = program.elementwise_sub(A, A);
  auto sub_2 = program.elementwise_sub(A, A);
  auto add_1 = program.add(B, sub_1);
  auto add_2 = program.add(B, sub_2);
  auto add   = program.add(add_1, add_2);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "CommonSubexpressionEliminationPass");
  auto scope = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  auto& prerun_instrs  = runtime_program->GetPreRunInstructions();
  auto& run_instrs     = runtime_program->GetRunInstructions();
  ASSERT_EQ(prerun_instrs.size(), 0);
  ASSERT_EQ(run_instrs.size(), 3);

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);

  runtime_program->Execute();
}

TEST(common_subexpression_elimination, common_subexpression_elimination_case3) {
  Placeholder A(Float(32), {32, 16}, "A");
  Placeholder B(Float(32), {32, 1}, "B", true);

  Program program;
  auto sub_1   = program.elementwise_sub(A, A);
  auto sub_2   = program.elementwise_sub(A, A);
  auto const_1 = program.fill_constant<float>({32, 16}, 1.0f, "", false, "const1");
  auto const_2 = program.fill_constant<float>({32, 16}, 1.0f, "", false, "const2");
  auto const_3 = program.fill_constant<float>({32, 16}, 2.0f, "", false, "const3");
  auto out1    = program.add(const_1, const_3);
  auto out2    = program.add(const_2, const_3);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "CommonSubexpressionEliminationPass");
  auto scope = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  auto& prerun_instrs  = runtime_program->GetPreRunInstructions();
  auto& run_instrs     = runtime_program->GetRunInstructions();
  ASSERT_EQ(prerun_instrs.size(), 0);
  ASSERT_EQ(run_instrs.size(), 4);

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);

  runtime_program->Execute();
  LOG(INFO) << "graph:\n" << graph->Visualize();
}

}  // namespace frontend
}  // namespace cinn

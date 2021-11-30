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

  auto c      = builder.elementwise_add(a, b, 1);
  auto d      = builder.relu(c);
  auto target = common::DefaultHostTarget();
  auto graph  = std::make_shared<Graph>(builder.Build(), target);

  // OpFusion will fuse add+relu, and the intermediate variable 'c' is eliminated
  ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  ASSERT_EQ(scope->var_names().size(), 4);
  EXPECT_NE(scope->FindVar(c->id), nullptr);

  GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  ASSERT_EQ(scope->var_names().size(), 3);
  EXPECT_EQ(scope->FindVar(c->id), nullptr);

  ASSERT_NO_THROW(runtime_program->Execute());
}

TEST(GraphCompilerTest, TestInsertBufferHandlers) {
  frontend::NetBuilder builder("test");
  auto a = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
  auto b = builder.CreateInput(Float(32), {64}, "B");

  auto c      = builder.elementwise_add(a, b, 1);
  auto d      = builder.relu(c);
  auto target = common::DefaultHostTarget();
  auto graph  = std::make_shared<Graph>(builder.Build(), target);
  ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);

  GraphCompiler gc_disable(target, scope, graph);
  GraphCompiler::CompileOptions options;
  // disable with_buffer_handle_instruction_inserted: only 1 instruction
  auto runtime_program_disable = gc_disable.Build(options).runtime_program;
  ASSERT_EQ(runtime_program_disable->size(), 1);
  const auto& computation_instr_disable = runtime_program_disable->GetRunInstructions().front();

  // enable with_buffer_handle_instruction_inserted: 3 instructions, 1st ->
  // malloc instruction(a, b, d), 2nd -> the real computation
  // instruction(add + relu)  and 3rd -> free instruction
  GraphCompiler gc_enable(target, scope, graph);
  options.with_buffer_handle_instruction_inserted = true;
  auto runtime_program_enable                     = gc_enable.Build(options).runtime_program;
  const auto& instructions                        = runtime_program_enable->GetRunInstructions();
  ASSERT_EQ(instructions.size(), 3);

  const auto& malloc_instr = instructions.front();
  ASSERT_EQ(malloc_instr->size(), 1);
  auto malloc_variable_names = malloc_instr->GetInArgs().front();
  auto used_variable_names   = std::unordered_set<std::string>(
      {static_cast<frontend::Variable>(a)->id, static_cast<frontend::Variable>(b)->id, d->id});
  EXPECT_EQ(malloc_instr->GetOutArgs().size(), 1);
  EXPECT_TRUE(malloc_instr->GetOutArgs().front().empty());
  EXPECT_EQ(malloc_variable_names.size(), 3);
  EXPECT_EQ(std::unordered_set<std::string>(malloc_variable_names.begin(), malloc_variable_names.end()),
            used_variable_names);

  const auto& computation_instr_enable = instructions.at(1);
  ASSERT_EQ(computation_instr_disable->size(), computation_instr_enable->size());
  EXPECT_EQ(computation_instr_disable->GetInArgs(), computation_instr_enable->GetInArgs());
  EXPECT_EQ(computation_instr_disable->GetOutArgs(), computation_instr_enable->GetOutArgs());

  const auto& free_instr = instructions.back();
  ASSERT_EQ(free_instr->size(), 1);
  EXPECT_EQ(free_instr->GetInArgs().size(), 1);
  EXPECT_TRUE(free_instr->GetInArgs().front().empty());
  auto free_variable_names = free_instr->GetOutArgs().front();
  EXPECT_EQ(std::unordered_set<std::string>(free_variable_names.begin(), free_variable_names.end()),
            used_variable_names);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

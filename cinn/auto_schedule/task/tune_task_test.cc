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

#include "cinn/auto_schedule/task/tune_task.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>

#include "cinn/auto_schedule/task/task_creator.h"
#include "cinn/common/context.h"
#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::frontend::NetBuilder;
using ::cinn::frontend::Program;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Scope;

Program CreateAddProgram() {
  constexpr int M = 32;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {M, N}, "A");
  auto b       = builder.CreateInput(Float(32), {M, N}, "B");
  auto c       = builder.Add(a, b);
  auto d       = builder.Add(a, c);
  auto program = builder.Build();

  return program;
}

TEST(TuneTask, GraphToUnoptLoweredFunc_NoPass) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target          = common::DefaultHostTarget();
#endif
  Program prog = CreateAddProgram();
  auto graph   = std::make_shared<hlir::framework::Graph>(prog, target);

  std::shared_ptr<cinn::hlir::framework::Scope> scope = BuildScope(target, graph);

  TaskCreator task_creator;
  std::vector<TuneTask> tasks = task_creator.CreateTuneTaskOpLevel(graph.get());

  GraphCompiler graph_compiler(target, scope, graph);

  ASSERT_EQ(tasks.size(), 2UL);

  std::stringstream ss;
  for (TuneTask& task : tasks) {
    task.SetGraphCompiler(&graph_compiler);
    task.TaskGraphToUnoptLoweredFunc();

    std::vector<ir::Expr> exprs = task.tune_context().GetLoweredFuncBodyExprs();
    VLOG(6) << "ir:Expr is: ";
    for (const ir::Expr& e : exprs) {
      VLOG(6) << e;
      ss << e << std::endl;
    }
  }

  std::string expr_str   = ss.str();
  std::string target_str = R"ROC(
{
  ScheduleBlock(root)
  {
    for (i, 0, 32)
    {
      for (j, 0, 24)
      {
        ScheduleBlock(elementwise_add_Out_0)
        {
          i0, i1 = axis.bind(i, j)
          elementwise_add_Out[i0, i1] = (A[i0, i1] + B[i0, i1])
        }
      }
    }
  }
}
{
  ScheduleBlock(root_0)
  {
    for (i, 0, 32)
    {
      for (j, 0, 24)
      {
        ScheduleBlock(elementwise_add_Out_1)
        {
          i0, i1 = axis.bind(i, j)
          elementwise_add_Out_1[i0, i1] = (A[i0, i1] + var_1[i0, i1])
        }
      }
    }
  }
}
)ROC";

  EXPECT_EQ(utils::Trim(target_str), utils::Trim(expr_str));
}

TEST(TuneTask, GraphToUnoptLoweredFunc_ApplyPass) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target          = common::DefaultHostTarget();
#endif
  Program prog = CreateAddProgram();
  auto graph   = std::make_shared<hlir::framework::Graph>(prog, target);
  ApplyPass(graph.get(), "OpFusion");

  std::shared_ptr<cinn::hlir::framework::Scope> scope = BuildScope(target, graph);

  TaskCreator task_creator;
  std::vector<TuneTask> tasks = task_creator.CreateTuneTaskOpLevel(graph.get());

  GraphCompiler graph_compiler(target, scope, graph);

  ASSERT_EQ(tasks.size(), 1UL);

  std::stringstream ss;
  for (TuneTask& task : tasks) {
    task.SetGraphCompiler(&graph_compiler);
    task.TaskGraphToUnoptLoweredFunc();

    std::vector<ir::Expr> exprs = task.tune_context().GetLoweredFuncBodyExprs();
    VLOG(6) << "ir:Expr is: ";
    for (const ir::Expr& e : exprs) {
      VLOG(6) << e;
      ss << e << std::endl;
    }
  }

  std::string expr_str = ss.str();
#ifdef CINN_WITH_CUDA
  std::string target_str = R"ROC(
{
  ScheduleBlock(root)
  {
    for (i, 0, 32)
    {
      for (j, 0, 24)
      {
        ScheduleBlock(elementwise_add_Out_1)
        {
          i0, i1 = axis.bind(i, j)
          elementwise_add_Out[i0, i1] = (A[i0, i1] + B[i0, i1])
        }
      }
    }
  }
}
{
  ScheduleBlock(root_0)
  {
    for (i, 0, 32)
    {
      for (j, 0, 24)
      {
        ScheduleBlock(elementwise_add_Out_0)
        {
          i0, i1 = axis.bind(i, j)
          elementwise_add_Out_0[i0, i1] = (A[i0, i1] + elementwise_add_Out[i0, i1])
        }
      }
    }
  }
}
)ROC";
#else
  std::string target_str = R"ROC(
{
  for (i, 0, 32)
  {
    for (j, 0, 24)
    {
      elementwise_add_Out[i, j] = (A[i, j] + B[i, j])
    }
  }
  for (i, 0, 32)
  {
    for (j, 0, 24)
    {
      elementwise_add_Out_0[i, j] = (A[i, j] + elementwise_add_Out[i, j])
    }
  }
}
)ROC";
#endif

  EXPECT_EQ(utils::Trim(target_str), utils::Trim(expr_str));
}

}  // namespace auto_schedule
}  // namespace cinn

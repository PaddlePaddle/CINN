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

#include <gtest/gtest.h>

#include <memory>

#include "cinn/auto_schedule/measure/schedule_measurer.h"
#include "cinn/auto_schedule/measure/simple_builder.h"
#include "cinn/auto_schedule/measure/simple_runner.h"
#include "cinn/auto_schedule/task/task_creator.h"
#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph_compiler.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;

frontend::Program CreateAddReluProgram() {
  constexpr int M = 32;
  constexpr int N = 24;
  frontend::NetBuilder builder("test");

  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Relu(c);
  return builder.Build();
}

TEST(ScheduleMeasurer, Basic) {
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  auto graph          = std::make_shared<Graph>(CreateAddReluProgram(), target);
  auto scope          = BuildScope(target, graph);
  auto graph_compiler = std::make_unique<GraphCompiler>(target, scope, graph);
  TaskCreator task_creator;
  std::vector<TuneTask> tasks = task_creator.CreateTuneTaskOpLevel(graph.get());
  ASSERT_EQ(2, tasks.size());

  std::vector<MeasureInput> inputs(tasks.size());
  for (int i = 0; i < tasks.size(); ++i) {
    auto* task              = &tasks[i];
    inputs[i].task          = task;
    inputs[i].lowered_funcs = graph_compiler->FusedGraphToLoweredFunc(task->task_graph());
  }

  auto builder                       = std::make_unique<SimpleBuilder>(graph_compiler.get());
  auto runner                        = std::make_unique<SimpleRunner>(1);
  auto measurer                      = std::make_unique<ScheduleMeasurer>(builder.get(), runner.get());
  std::vector<MeasureResult> results = measurer->Measure(inputs);
  ASSERT_EQ(inputs.size(), results.size());
}

}  // namespace auto_schedule
}  // namespace cinn

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

#include "cinn/auto_schedule/auto_tuner.h"

#include <gtest/gtest.h>

#include <iostream>

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
  frontend::NetBuilder builder("test");

  auto a = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
  auto b = builder.CreateInput(Float(32), {64}, "B");
  auto c = builder.ElementwiseAdd(a, b, 1);
  auto d = builder.Relu(c);

  return builder.Build();
}

TEST(AutoTuner, ZeroMeasure) {
  auto target         = common::DefaultHostTarget();
  auto graph          = std::make_shared<Graph>(CreateAddReluProgram(), target);
  auto scope          = BuildScope(target, graph);
  auto graph_compiler = std::make_unique<GraphCompiler>(target, scope, graph);

  auto tuner = std::make_unique<AutoTuner>(target, graph.get());
  AutoTuner::Config tuning_config;
  tuning_config.task_schedule_strategy = "round_robin";
  tuner->Initialize(tuning_config, graph_compiler.get());

  TuningOptions tuning_options;
  tuning_options.num_tuning_rounds         = 1;
  tuning_options.num_measure_trials        = 0;
  tuning_options.num_samples_per_iteration = 10;
  TuningResult result                      = tuner->Tune(tuning_options);

  ASSERT_EQ(2, result.tuned_graphs.size());
  const auto& sub_graph1 = result.tuned_graphs.front();
  ASSERT_EQ(1, sub_graph1.groups.size());
  ASSERT_EQ(sub_graph1.groups[0][0]->op()->name, "elementwise_add");
  const auto& sub_graph2 = result.tuned_graphs.back();
  ASSERT_EQ(1, sub_graph2.groups.size());
  ASSERT_EQ(sub_graph2.groups[0][0]->op()->name, "relu");

  ASSERT_EQ(result.optimized_lowered.size(), 2UL);
  ASSERT_EQ(result.optimized_lowered[0].lowered_funcs.size(), 1UL);
  ASSERT_EQ(result.optimized_lowered[0].lowered_funcs[0].size(), 1UL);
}

TEST(AutoTuner, NonZeroMeasure) {
  auto target         = common::DefaultHostTarget();
  auto graph          = std::make_shared<Graph>(CreateAddReluProgram(), target);
  auto scope          = BuildScope(target, graph);
  auto graph_compiler = std::make_unique<GraphCompiler>(target, scope, graph);

  auto tuner = std::make_unique<AutoTuner>(target, graph.get());
  AutoTuner::Config tuning_config;
  tuning_config.task_schedule_strategy = "round_robin";
  tuner->Initialize(tuning_config, graph_compiler.get());

  TuningOptions tuning_options;
  tuning_options.num_tuning_rounds         = 1;
  tuning_options.num_measure_trials        = 30;
  tuning_options.num_samples_per_iteration = 10;
  TuningResult result                      = tuner->Tune(tuning_options);

  ASSERT_EQ(2, result.tuned_graphs.size());
  const auto& sub_graph1 = result.tuned_graphs.front();
  ASSERT_EQ(1, sub_graph1.groups.size());
  ASSERT_EQ(sub_graph1.groups[0][0]->op()->name, "elementwise_add");
  const auto& sub_graph2 = result.tuned_graphs.back();
  ASSERT_EQ(1, sub_graph2.groups.size());
  ASSERT_EQ(sub_graph2.groups[0][0]->op()->name, "relu");

  ASSERT_EQ(result.optimized_lowered.size(), 2UL);
  ASSERT_EQ(result.optimized_lowered[0].lowered_funcs.size(), 1UL);
  ASSERT_EQ(result.optimized_lowered[0].lowered_funcs[0].size(), 1UL);
}

}  // namespace auto_schedule
}  // namespace cinn

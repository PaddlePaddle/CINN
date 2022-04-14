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
using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;

class TestAutoTuner : public ::testing::Test {
 public:
  // TODO(CtfGo): add test on GPU once the bug of thread.idx not bound is fixed
  Target target = common::DefaultHostTarget();

  std::shared_ptr<Graph> graph;
  std::shared_ptr<Scope> compiled_scope;
  std::unique_ptr<GraphCompiler> graph_compiler;
  std::unique_ptr<AutoTuner> tuner;

  static frontend::Program CreateAddReluProgram();
  void SetUp() override {
    graph          = std::make_shared<Graph>(CreateAddReluProgram(), target);
    compiled_scope = BuildScope(target, graph);
    graph_compiler = std::make_unique<GraphCompiler>(target, compiled_scope, graph);
    tuner          = std::make_unique<AutoTuner>(target, graph.get());
  }

  TuningResult InitializeAndTune(const AutoTuner::Config& config, const TuningOptions& options) {
    tuner->Initialize(config, graph_compiler.get());
    return tuner->Tune(options);
  }

  void BasicCheckResult(const TuningResult& result) {
    ASSERT_EQ(2, result.tuned_graph.size());
    const auto& sub_graph1 = result.tuned_graph.front();
    ASSERT_EQ(1, sub_graph1.groups.size());
    ASSERT_EQ(sub_graph1.groups[0][0]->op()->name, "elementwise_add");
    const auto& sub_graph2 = result.tuned_graph.back();
    ASSERT_EQ(1, sub_graph2.groups.size());
    ASSERT_EQ(sub_graph2.groups[0][0]->op()->name, "relu");

    ASSERT_EQ(result.optimized_exprs.size(), 2UL);
    ASSERT_EQ(result.optimized_exprs[0].lowered_funcs.size(), 1UL);
    ASSERT_EQ(result.optimized_exprs[0].lowered_funcs[0].size(), 1UL);
  }

  void ApplyTunedAndRun(const TuningResult& result) {
    // build runtime program with tuning result
    GraphCompiler::CompileOptions compile_options;
    compile_options.with_instantiate_variables = true;
    compile_options.Apply(result);
    ASSERT_EQ(2, compile_options.groups.size());
    ASSERT_EQ(2, compile_options.lowered_funcs.size());

    auto runtime_program = graph_compiler->Build(compile_options).runtime_program;
    ASSERT_EQ(2, runtime_program->size());
    runtime_program->Execute();
  }
};

frontend::Program TestAutoTuner::CreateAddReluProgram() {
  frontend::NetBuilder builder("test");

  auto a = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
  auto b = builder.CreateInput(Float(32), {64}, "B");
  auto c = builder.ElementwiseAdd(a, b, 1);
  auto d = builder.Relu(c);

  return builder.Build();
}

TEST_F(TestAutoTuner, ZeroMeasure) {
  // set config and options
  AutoTuner::Config tuning_config;
  tuning_config.task_schedule_strategy = "round_robin";

  TuningOptions tuning_options;
  tuning_options.num_measure_trials = 0;
  auto result                       = InitializeAndTune(tuning_config, tuning_options);
  BasicCheckResult(result);
  ApplyTunedAndRun(result);
}

TEST_F(TestAutoTuner, NonZeroMeasure) {
  // set config and options
  AutoTuner::Config tuning_config;
  tuning_config.task_schedule_strategy = "round_robin";

  TuningOptions tuning_options;
  tuning_options.num_measure_trials        = 4;
  tuning_options.num_samples_per_iteration = 2;

  auto result = InitializeAndTune(tuning_config, tuning_options);
  BasicCheckResult(result);
  ApplyTunedAndRun(result);
}

}  // namespace auto_schedule
}  // namespace cinn

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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>

#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/ir/ir_base.h"
#include "cinn/utils/data_util.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;

class PerformanceTester {
 public:
  virtual frontend::Program CreateProgram() = 0;
  virtual void PrepareData(std::shared_ptr<Scope> scope) = 0;

  void PrepareData() {
    PrepareData(no_schedule_compiled_scope_);
    PrepareData(manual_schedule_compiled_scope_);
    PrepareData(auto_schedule_compiled_scope_);
  }
  
  void BuildRuntimePrograms() {
    BuildNoScheduleProgram();
    BuildManualScheduleProgram();
    BuildAutoScheduleProgram();
  }

  void Run(int repeat) {
    // TODO  no_schedule_program_->ExecuteTest(repeat);
    manual_schedule_program_->ExecuteTest(repeat);
    auto_schedule_program_->ExecuteTest(repeat);
  }

  void BuildAndRun(int repeat) {
    auto program = CreateProgram();
    graph_ = std::make_shared<hlir::framework::Graph>(program, target_);
    VLOG(3) << "Initialize graph completed, start building runtime program.";
    BuildRuntimePrograms();
    VLOG(3) << "Build runtime programs completed, start preparing data.";
    PrepareData();
    VLOG(3) << "Prepare data completed, start running.";
    Run(repeat);
  }

 protected:
  void BuildNoScheduleProgram() {
    LOG(INFO) << "not implemented.";
    // TODO  Add no schedule build process.
    no_schedule_compiled_scope_ = BuildScope(target_, graph_);
    no_schedule_graph_compiler_ = std::make_unique<GraphCompiler>(target_, no_schedule_compiled_scope_, graph_);
  }

  void BuildManualScheduleProgram() {
    manual_schedule_compiled_scope_ = BuildScope(target_, graph_);
    manual_schedule_graph_compiler_ = std::make_unique<GraphCompiler>(target_, manual_schedule_compiled_scope_, graph_);
    manual_schedule_program_ = manual_schedule_graph_compiler_->Build();
  }

  void BuildAutoScheduleProgram() {
    auto_schedule_compiled_scope_ = BuildScope(target_, graph_);
    auto_schedule_graph_compiler_ = std::make_unique<GraphCompiler>(target_, auto_schedule_compiled_scope_, graph_);
    tuner_ = std::make_unique<AutoTuner>(target_, graph_.get());

    AutoTuner::Config tuning_config;
    tuning_config.task_schedule_strategy = "round_robin";

    TuningOptions tuning_options;
    tuning_options.num_measure_trials = 0;

    tuner_->Initialize(tuning_config, auto_schedule_graph_compiler_.get());
    TuningResult tuning_result = tuner_->Tune(tuning_options);

    GraphCompiler::CompileOptions compile_options;
    compile_options.with_instantiate_variables = true;
    compile_options.Apply(tuning_result);

    auto_schedule_program_ = auto_schedule_graph_compiler_->Build(compile_options).runtime_program;
  }

#ifdef CINN_WITH_CUDA
  Target target_ = common::DefaultNVGPUTarget();
#else
  Target target_ = common::DefaultHostTarget();
#endif

  std::shared_ptr<Graph> graph_;

  std::shared_ptr<Scope> no_schedule_compiled_scope_;
  std::shared_ptr<Scope> manual_schedule_compiled_scope_;
  std::shared_ptr<Scope> auto_schedule_compiled_scope_;

  std::unique_ptr<GraphCompiler> no_schedule_graph_compiler_;
  std::unique_ptr<GraphCompiler> manual_schedule_graph_compiler_;
  std::unique_ptr<GraphCompiler> auto_schedule_graph_compiler_;

  std::unique_ptr<hlir::framework::Program> no_schedule_program_;
  std::unique_ptr<hlir::framework::Program> manual_schedule_program_;
  std::unique_ptr<hlir::framework::Program> auto_schedule_program_;
  
  std::unique_ptr<AutoTuner> tuner_;
};

class MatmulPerformanceTester : public PerformanceTester {
 public:
  MatmulPerformanceTester(int M, int K, int N) : M_(M), K_(K), N_(N) {}

  frontend::Program CreateProgram() override {
    frontend::NetBuilder builder("matmul_net_builder");
    auto x = builder.CreateInput(Float(32), {M_, K_}, "X");
    auto y = builder.CreateInput(Float(32), {N_, K_}, "Y");

    auto mul_out = builder.Mul(x, y, 1, 1);
    return builder.Build();
  }
  void PrepareData(std::shared_ptr<Scope> scope) override {
    scope->Var<hlir::framework::Tensor>("X");
    scope->Var<hlir::framework::Tensor>("Y");
    auto x_tensor        = scope->GetTensor("X");
    auto y_tensor        = scope->GetTensor("Y");
    SetRandData<float>(x_tensor, target_);
    SetRandData<float>(y_tensor, target_);
  }

 private:
  int M_;
  int K_;
  int N_;
};

class AddPerformanceTester : public PerformanceTester {
 public:
  AddPerformanceTester(int M, int N) : M_(M), N_(N) {}

  frontend::Program CreateProgram() override {
    frontend::NetBuilder builder("add_net_builder");
    auto x = builder.CreateInput(Float(32), {M_, N_}, "X");
    auto y = builder.CreateInput(Float(32), {M_, N_}, "Y");

    auto mul_out = builder.Add(x, y);
    return builder.Build();
  }
  void PrepareData(std::shared_ptr<Scope> scope) override {
    scope->Var<hlir::framework::Tensor>("X");
    scope->Var<hlir::framework::Tensor>("Y");
    auto x_tensor        = scope->GetTensor("X");
    auto y_tensor        = scope->GetTensor("Y");
    SetRandData<float>(x_tensor, target_);
    SetRandData<float>(y_tensor, target_);
  }

 private:
  int M_;
  int N_;
};

TEST(MatmulPerformanceTest, matmul_32x16x32) {
  int M = 32;
  int K = 16;
  int N = 32;
  MatmulPerformanceTester tester(M, K, N);
  tester.BuildAndRun(100);
}

TEST(MatmulPerformanceTest, matmul_1024x1024x1024) {
  int M = 1024;
  int K = 1024;
  int N = 1024;
  MatmulPerformanceTester tester(M, K, N);
  tester.BuildAndRun(100);
}

TEST(AddPerformanceTest, add_32x16) {
  int M = 32;
  int N = 16;
  AddPerformanceTester tester(M, N);
  tester.BuildAndRun(100);
}

}  // namespace auto_schedule
}  // namespace cinn

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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>

#include "cinn/auto_schedule/auto_tuner.h"
#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/optimize.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/ir/ir_base.h"
#include "cinn/utils/data_util.h"

DEFINE_string(resnet50_model_dir, "", "");

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

  void BuildRuntimePrograms(int num_tuning_rounds) {
    scope_          = BuildScope(target_, graph_, scope_);
    graph_compiler_ = std::make_unique<GraphCompiler>(target_, scope_, graph_);
    BuildNoScheduleProgram();
    BuildManualScheduleProgram();
    BuildAutoScheduleProgram(num_tuning_rounds);
  }

  void Run(int repeat) {
    no_schedule_program_->ExecuteTest(repeat);
    manual_schedule_program_->ExecuteTest(repeat);
    auto_schedule_program_->ExecuteTest(repeat);
  }

  void BuildAndRun(int repeat, int num_tuning_rounds) {
    auto program = CreateProgram();
    graph_       = std::make_shared<hlir::framework::Graph>(program, target_);
    // graph_ = frontend::Optimize(&program, {}, target_);
    hlir::framework::ApplyPass(graph_.get(), "InferShape");
    VLOG(3) << "Initialize graph completed, start building runtime program.";
    BuildRuntimePrograms(num_tuning_rounds);
    VLOG(3) << "Build runtime programs completed, start running.";
    Run(repeat);
  }

 protected:
  void BuildNoScheduleProgram() {
    std::tuple<std::vector<common::GraphNode*>, std::vector<common::GraphEdge*>> topo_result =
        graph_->topological_order();
    const std::vector<common::GraphNode*>& nodes = std::get<0>(topo_result);
    std::vector<std::vector<hlir::framework::Node*>> task_graph;
    for (common::GraphNode* n : nodes) {
      hlir::framework::Node* op_node = n->safe_as<hlir::framework::Node>();
      if (op_node) {
        task_graph.push_back(std::vector<hlir::framework::Node*>(1, op_node));
      }
    }
    std::vector<std::vector<ir::LoweredFunc>> funcs = graph_compiler_->FusedGraphToLoweredFunc(task_graph);

    GraphCompiler::CompileOptions compile_options;
    compile_options.with_instantiate_variables = true;
    compile_options.groups                     = task_graph;
    compile_options.lowered_funcs              = funcs;

    VLOG(3) << "===========================No Schedule LoweredFunc Begin===========================";
    for (const auto& funcvec : funcs) {
      for (const auto& func : funcvec) {
        VLOG(3) << func;
      }
    }
    VLOG(3) << "===========================No Schedule LoweredFunc End=============================";

    no_schedule_program_ = graph_compiler_->Build(compile_options).runtime_program;
  }

  void BuildManualScheduleProgram() {
    manual_schedule_program_ = graph_compiler_->Build();

    VLOG(3) << "===========================Manual Schedule LoweredFunc Begin===========================";
    graph_compiler_->PrintFunc();
    VLOG(3) << "===========================Manual Schedule LoweredFunc End=============================";
  }

  void BuildAutoScheduleProgram(int num_tuning_rounds = 10) {
    tuner_ = std::make_unique<AutoTuner>(target_, graph_.get());

    AutoTuner::Config tuning_config;
    TuningOptions tuning_options;
    tuning_options.num_tuning_rounds = num_tuning_rounds;

    tuner_->Initialize(tuning_config, graph_compiler_.get());
    TuningResult tuning_result = tuner_->Tune(tuning_options);

    GraphCompiler::CompileOptions compile_options;
    compile_options.with_instantiate_variables = true;
    compile_options.Apply(tuning_result);

    VLOG(3) << "===========================Auto Schedule LoweredFunc Begin===========================";
    for (const auto& funcvec : compile_options.lowered_funcs) {
      for (const auto& func : funcvec) {
        VLOG(3) << func;
      }
    }
    VLOG(3) << "===========================Auto Schedule LoweredFunc End=============================";

    auto_schedule_program_ = graph_compiler_->Build(compile_options).runtime_program;
  }

#ifdef CINN_WITH_CUDA
  Target target_ = common::DefaultNVGPUTarget();
#else
  Target target_ = common::DefaultHostTarget();
#endif

  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Scope> scope_;
  std::unique_ptr<GraphCompiler> graph_compiler_;

  std::unique_ptr<hlir::framework::Program> no_schedule_program_;
  std::unique_ptr<hlir::framework::Program> manual_schedule_program_;
  std::unique_ptr<hlir::framework::Program> auto_schedule_program_;

  std::unique_ptr<AutoTuner> tuner_;
};

class MulPerformanceTester : public PerformanceTester {
 public:
  explicit MulPerformanceTester(int M, int K, int N) : M_(M), K_(K), N_(N) {}

  frontend::Program CreateProgram() override {
    frontend::NetBuilder builder("mul_net_builder");
    auto x = builder.CreateInput(Float(32), {M_, K_}, "X");
    auto y = builder.CreateInput(Float(32), {N_, K_}, "Y");

    auto mul_out = builder.Mul(x, y, 1, 1);
    return builder.Build();
  }

 private:
  int M_;
  int K_;
  int N_;
};

class MatmulPerformanceTester : public PerformanceTester {
 public:
  explicit MatmulPerformanceTester(int M, int K, int N) : M_(M), K_(K), N_(N) {}

  frontend::Program CreateProgram() override {
    frontend::NetBuilder builder("mul_net_builder");
    auto x = builder.CreateInput(Float(32), {M_, K_}, "X");
    auto y = builder.CreateInput(Float(32), {K_, N_}, "Y");

    auto mul_out = builder.Matmul(x, y);
    return builder.Build();
  }

 private:
  int M_;
  int K_;
  int N_;
};

class AddPerformanceTester : public PerformanceTester {
 public:
  explicit AddPerformanceTester(int M, int N) : M_(M), N_(N) {}

  frontend::Program CreateProgram() override {
    frontend::NetBuilder builder("add_net_builder");
    auto x = builder.CreateInput(Float(32), {M_, N_}, "X");
    auto y = builder.CreateInput(Float(32), {M_, N_}, "Y");

    auto mul_out = builder.Add(x, y);
    return builder.Build();
  }

 private:
  int M_;
  int N_;
};

class PaddleModelPerformanceTester : public PerformanceTester {
 public:
  explicit PaddleModelPerformanceTester(const std::string& model_path,
                                        const std::vector<std::string>& input_names,
                                        const std::vector<std::vector<int>>& input_shapes)
      : model_path_(model_path), input_names_(input_names), input_shapes_(input_shapes) {}

  frontend::Program CreateProgram() override {
    CHECK(!input_names_.empty());
    CHECK_EQ(input_names_.size(), input_shapes_.size());

    scope_             = std::make_shared<hlir::framework::Scope>();
    auto loadedProgram = cinn::frontend::LoadPaddleProgram(model_path_, scope_.get(), true, target_);
    auto& program      = std::get<0>(loadedProgram);
    auto& varmap       = std::get<1>(loadedProgram);
    VLOG(3) << "loaded program: " << *program;
    CHECK(!varmap.empty());

    std::vector<frontend::Variable> input_vars;
    std::transform(input_names_.begin(), input_names_.end(), std::back_inserter(input_vars), [&](const std::string& x) {
      return varmap.at(x);
    });

    for (int i = 0; i < input_vars.size(); i++) {
      input_vars[i]->shape = input_shapes_[i];
    }

    program->SetInputs(input_vars);
    program->Validate();

    return *program;
  }

 private:
  const std::string model_path_;
  std::vector<std::string> input_names_;
  std::vector<std::vector<int>> input_shapes_;
};

#ifdef CINN_WITH_CUDA

const int repeat_time       = 100;
const int num_tuning_rounds = 1;

TEST(MatmulPerformanceTest, matmul_32x16x32) {
  int M = 32;
  int K = 16;
  int N = 32;
  MatmulPerformanceTester tester(M, K, N);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

TEST(MulPerformanceTest, mul_32x16x32) {
  int M = 32;
  int K = 16;
  int N = 32;
  MulPerformanceTester tester(M, K, N);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

TEST(AddPerformanceTest, add_32x16) {
  int M = 32;
  int N = 16;
  AddPerformanceTester tester(M, N);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

TEST(AddPerformanceTest, add_1024x1024) {
  int M = 1024;
  int N = 1024;
  AddPerformanceTester tester(M, N);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

TEST(PaddleModelPerformanceTester, ResNet50) {
  std::vector<std::string> input_names       = {"inputs"};
  std::vector<std::vector<int>> input_shapes = {{1, 3, 224, 224}};
  PaddleModelPerformanceTester tester(FLAGS_resnet50_model_dir, input_names, input_shapes);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

#endif

}  // namespace auto_schedule
}  // namespace cinn

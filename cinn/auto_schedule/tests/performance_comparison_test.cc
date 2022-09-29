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

#include <bitset>
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
#include "cinn/runtime/flags.h"
#include "cinn/utils/data_util.h"

DEFINE_string(resnet50_model_dir, "./ResNet50", "the path to paddle model resnet50.");
DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;

class ProgramBuilder {
  ProgramBuilder();
  virtual frontend::Program operator()() = 0;
};

class PerformanceTester {
 public:
  virtual frontend::Program CreateProgram() = 0;

  void BuildRuntimePrograms(int num_tuning_rounds) {
    scope_          = BuildScope(target_, graph_, scope_);
    graph_compiler_ = std::make_unique<GraphCompiler>(target_, scope_, graph_);
    if (option_flags_.test(0)) {
      VLOG(3) << "Build no schedule program.";
      BuildNoScheduleProgram();
    }
    if (option_flags_.test(1)) {
      VLOG(3) << "Build manual schedule program.";
      BuildManualScheduleProgram();
    }
    if (option_flags_.test(2)) {
      VLOG(3) << "Build auto schedule program.";
      BuildAutoScheduleProgram(num_tuning_rounds);
    }
  }

  void Run(int repeat) {
    if (option_flags_.test(0)) {
      VLOG(3) << "Execute no schedule program.";
      no_schedule_program_->ExecuteTest(repeat);
    }
    if (option_flags_.test(1)) {
      VLOG(3) << "Execute manual schedule program.";
      manual_schedule_program_->ExecuteTest(repeat);
    }
    if (option_flags_.test(2)) {
      VLOG(3) << "Execute auto schedule program.";
      auto_schedule_program_->ExecuteTest(repeat);
    }
  }

  void BuildAndRun(int repeat, int num_tuning_rounds, bool initialize_graph = false) {
    if (initialize_graph || !is_graph_initialized_) {
      VLOG(3) << "Start initialize graph.";
      auto program = CreateProgram();
      graph_       = std::make_shared<hlir::framework::Graph>(program, target_);
      // hlir::framework::ApplyPass(graph_.get(), "InferShape");
      is_graph_initialized_ = true;
      VLOG(3) << "Initialize graph completed.";
    }

    VLOG(3) << "start building runtime program.";
    BuildRuntimePrograms(num_tuning_rounds);
    VLOG(3) << "Build runtime programs completed, start running.";
    Run(repeat);
  }

  void SetOptionFlags(unsigned long options) {
    CHECK_LE(options, 7UL) << "options can not be greater than 7";
    option_flags_ = options;
  }

 protected:
  void BuildNoScheduleProgram() {
    const auto& dtype_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, common::Type>>("inferdtype");
    const auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");

    std::shared_ptr<hlir::framework::OpLowerer> op_lowerer =
        std::make_unique<hlir::framework::OpLowerer>(dtype_dict, shape_dict, target_);

    GraphCompiler::CompileOptions compile_options;
    compile_options.with_instantiate_variables = true;

    std::tuple<std::vector<common::GraphNode*>, std::vector<common::GraphEdge*>> topo_result =
        graph_->topological_order();
    const std::vector<common::GraphNode*>& nodes_in_order = std::get<0>(topo_result);
    for (auto graph_node : nodes_in_order) {
      // n must be an op node
      auto node = graph_node->safe_as<hlir::framework::Node>();
      if (node) {
        auto group = std::make_shared<Graph::Group>();
        // init group
        group->nodes.push_back(node);
        group->nodes_set.insert(node);
        group->output_nodes.insert(node);
        // input node
        for (auto& edge : node->inlinks()) {
          auto input_graph_node = edge->source();
          auto input_node_data  = input_graph_node->safe_as<hlir::framework::NodeData>();
          CHECK(input_node_data);
          // input data has no source node
          if (input_node_data->source_node.get()) {
            group->input_nodes[input_node_data->source_node.get()] = 1;
          }
        }

        // group type
        group->op_pattern_kind = hlir::framework::kOpaque;
        // use current node as master node for schedule
        group->master_nodes.insert(node);
        group->group_id = node->id();

        compile_options.groups.push_back(group);
        compile_options.lowered_funcs.push_back(op_lowerer->LowerWithoutSchedule(group));
      }
    }

    VLOG(3) << "===========================No Schedule LoweredFunc Begin===========================";
    for (const auto& funcvec : compile_options.lowered_funcs) {
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
    // graph_compiler_->PrintFunc();
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

  // Flags that control which schedule tests will be run.
  // Bit with index 0 controls no schedule test, means options = 1 = "001" will run no schedule test.
  // Bit with index 1 controls manual schedule test, means options = 2 = "010" will run manual schedule test.
  // Bit with index 2 controls auto schedule test, means options = 4 = "100" will run auto schedule test.
  // The default value is 7, which means that all tests will be run.
  std::bitset<3> option_flags_ = 7UL;
  bool is_graph_initialized_   = false;
};

class MulPerformanceTester : public PerformanceTester {
 public:
  MulPerformanceTester(int M, int K, int N) : M_(M), K_(K), N_(N) {}

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
  MatmulPerformanceTester(int M, int K, int N) : M_(M), K_(K), N_(N) {}

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
  AddPerformanceTester(int M, int N) : M_(M), N_(N) {}

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
  PaddleModelPerformanceTester(const std::string& model_path,
                               const std::vector<std::string>& input_names,
                               const std::vector<std::vector<int>>& input_shapes)
      : model_path_(model_path), input_names_(input_names), input_shapes_(input_shapes) {}

  frontend::Program CreateProgram() override {
    CHECK(!input_names_.empty());
    CHECK_EQ(input_names_.size(), input_shapes_.size());

    scope_ = std::make_shared<hlir::framework::Scope>();
    std::unordered_map<std::string, std::vector<int>> input_to_shape;
    for (int idx = 0; idx < input_names_.size(); ++idx) {
      input_to_shape[input_names_[idx]] = input_shapes_[idx];
    }
    auto loadedProgram = cinn::frontend::LoadPaddleProgram(model_path_, scope_.get(), input_to_shape, true, target_);
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
  FLAGS_cinn_ir_schedule = true;
  int M                  = 32;
  int K                  = 16;
  int N                  = 32;
  MatmulPerformanceTester tester(M, K, N);
  // tester.SetOptionFlags(5UL);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

TEST(MulPerformanceTest, mul_32x16x32) {
  FLAGS_cinn_ir_schedule = true;
  int M                  = 32;
  int K                  = 16;
  int N                  = 32;
  MulPerformanceTester tester(M, K, N);
  // tester.SetOptionFlags(5UL);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

TEST(AddPerformanceTest, add_32x16) {
  FLAGS_cinn_ir_schedule = true;
  int M                  = 32;
  int N                  = 16;
  AddPerformanceTester tester(M, N);
  // tester.SetOptionFlags(5UL);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

TEST(PaddleModelPerformanceTester, ResNet50) {
  FLAGS_cinn_ir_schedule                     = true;
  std::vector<std::string> input_names       = {"inputs"};
  std::vector<std::vector<int>> input_shapes = {{1, 3, 224, 224}};

  CHECK_NE(FLAGS_resnet50_model_dir, "");
  // ResNet50 can only run manual schedule test now.
  PaddleModelPerformanceTester tester(FLAGS_resnet50_model_dir, input_names, input_shapes);
  tester.SetOptionFlags(4UL);
  tester.BuildAndRun(repeat_time, num_tuning_rounds);
}

#endif

}  // namespace auto_schedule
}  // namespace cinn

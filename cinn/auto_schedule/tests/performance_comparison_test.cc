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
#include "cinn/auto_schedule/tests/paddle_model_program_builder.h"
#include "cinn/auto_schedule/tests/single_op_program_builder.h"
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

class PerformanceTester : public ::testing::Test {
 public:
  void SetUp() override {
    // AutoTuner is combined with new IR Schedule
    FLAGS_cinn_ir_schedule = true;
  }

  void BuildRuntimePrograms(int num_tuning_rounds) {
    scope_          = BuildScope(target_, graph_);
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

  void BuildAndRun(int repeat, int num_tuning_rounds, const frontend::Program& program) {
    VLOG(3) << "Start initialize graph.";
    graph_ = std::make_shared<hlir::framework::Graph>(program, target_);
    hlir::framework::ApplyPass(graph_.get(), "InferShape");
    VLOG(3) << "Initialize graph completed, Start building runtime program.";
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

  void BuildManualScheduleProgram() { manual_schedule_program_ = graph_compiler_->Build(); }

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
};

#ifdef CINN_WITH_CUDA

const int repeat_time       = 100;
const int num_tuning_rounds = 1;
const int batch_size        = 1;

TEST_F(PerformanceTester, Mul) {
  int M = 32;
  int K = 16;
  int N = 32;
  SetOptionFlags(7UL);
  BuildAndRun(repeat_time, num_tuning_rounds, MulProgramBuilder({M, K}, {N, K})());
}

TEST_F(PerformanceTester, Add) {
  SetOptionFlags(7UL);
  BuildAndRun(repeat_time, num_tuning_rounds, AddProgramBuilder({1, 56, 56, 256}, {1, 56, 56, 256})());
}

TEST_F(PerformanceTester, Matmul) {
  int M = batch_size;
  int K = 2048;
  int N = 1000;
  SetOptionFlags(7UL);
  BuildAndRun(repeat_time, num_tuning_rounds, MatmulProgramBuilder({M, K}, {K, N})());
}

TEST_F(PerformanceTester, Relu) {
  SetOptionFlags(7UL);
  BuildAndRun(repeat_time, num_tuning_rounds, ReluProgramBuilder({batch_size, 64, 56, 56})());
}

TEST_F(PerformanceTester, Conv2d) {
  std::vector<int32_t> input_shape{batch_size, 3, 224, 224};
  std::vector<int32_t> weight_shape{64, 3, 7, 7};
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{3, 3};
  std::vector<int> dilations{1, 1};
  int groups                    = 1;
  std::string data_format       = "NCHW";
  std::string padding_algorithm = "EXPLICIT";

  SetOptionFlags(0UL);
  BuildAndRun(repeat_time,
              num_tuning_rounds,
              Conv2dProgramBuilder(
                  input_shape, weight_shape, strides, paddings, dilations, groups, data_format, padding_algorithm)());
}

TEST_F(PerformanceTester, Pool2d) {
  std::vector<int32_t> input_shape{batch_size, 64, 112, 112};
  std::string pooling_type = "max";
  std::vector<int> ksize{3, 3};
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{1, 1};
  bool ceil_mode                = false;
  bool exclusive                = true;
  bool global_pooling           = false;
  std::string data_format       = "NCHW";
  bool adaptive                 = false;
  std::string padding_algorithm = "EXPLICIT";

  SetOptionFlags(0UL);
  BuildAndRun(repeat_time,
              num_tuning_rounds,
              Pool2dProgramBuilder(input_shape,
                                   pooling_type,
                                   ksize,
                                   strides,
                                   paddings,
                                   ceil_mode,
                                   exclusive,
                                   global_pooling,
                                   data_format,
                                   adaptive,
                                   padding_algorithm)());
}

TEST_F(PerformanceTester, BatchNorm) {
  std::vector<int32_t> input_shape{batch_size, 64, 112, 112};
  std::vector<int32_t> scale_shape{64};
  std::vector<int32_t> bias_shape{64};
  std::vector<int32_t> mean_shape{64};
  std::vector<int32_t> variance_shape{64};
  float epsilon                  = 1e-5f;
  float momentum                 = 0.9f;
  const std::string& data_layout = "NCHW";
  bool is_test                   = true;

  SetOptionFlags(7UL);
  BuildAndRun(
      repeat_time,
      num_tuning_rounds,
      BatchNormProgramBuilder(
          input_shape, scale_shape, bias_shape, mean_shape, variance_shape, epsilon, momentum, data_layout, is_test)());
}

TEST_F(PerformanceTester, Reshape) {
  std::vector<int32_t> input_shape{batch_size, 2048, 1, 1};
  std::vector<int32_t> output_shape{batch_size, 2048};

  SetOptionFlags(7UL);
  BuildAndRun(repeat_time, num_tuning_rounds, ReshapeProgramBuilder(input_shape, output_shape)());
}

TEST_F(PerformanceTester, Softmax) {
  std::vector<int32_t> input_shape{batch_size, 1000};
  int axis                = -1;
  std::string data_format = "AnyLayout";

  SetOptionFlags(0UL);
  BuildAndRun(repeat_time, num_tuning_rounds, SoftmaxProgramBuilder(input_shape, axis, data_format)());
}

TEST_F(PerformanceTester, Scale) {
  std::vector<int32_t> input_shape{batch_size, 1000};
  float scale           = 1.0f;
  float bias            = 0.0f;
  bool bias_after_scale = true;

  SetOptionFlags(7UL);
  BuildAndRun(repeat_time, num_tuning_rounds, ScaleProgramBuilder(input_shape, scale, bias, bias_after_scale)());
}

// paddle model test
TEST_F(PerformanceTester, ResNet50) {
  std::vector<std::string> input_names       = {"inputs"};
  std::vector<std::vector<int>> input_shapes = {{batch_size, 3, 224, 224}};
  CHECK_NE(FLAGS_resnet50_model_dir, "");
  SetOptionFlags(0UL);
  BuildAndRun(
      repeat_time, num_tuning_rounds, PaddleModelProgramBuilder(FLAGS_resnet50_model_dir, input_names, input_shapes)());
}

#endif

}  // namespace auto_schedule
}  // namespace cinn

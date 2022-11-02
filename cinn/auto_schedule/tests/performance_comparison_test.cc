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
#include "cinn/hlir/pass/op_fusion_pass.h"
#include "cinn/ir/ir_base.h"
#include "cinn/runtime/flags.h"
#include "cinn/utils/data_util.h"

DEFINE_string(resnet50_model_dir, "./ResNet50", "the path to paddle model resnet50.");
// Flags that control which schedule tests will be run.
// Bit with index 0 controls no schedule test, means options = 1 = "001" will run no schedule test.
// Bit with index 1 controls manual schedule test, means options = 2 = "010" will run manual schedule test.
// Bit with index 2 controls auto schedule test, means options = 4 = "100" will run auto schedule test.
// The default value is -1, which means that this flag is disabled to set the options
DEFINE_int32(evaluate_knobs, -1, "the options to control which schedule tests will be run.");
DECLARE_int32(cinn_parallel_compile_size);

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;

class PerformanceTester : public ::testing::Test {
 public:
  struct Options {
    // times of compiled runtime program will be executed repeatedly.
    int repeat_times = 2;
    // the num_tuning_rounds for auto tuning
    int num_tuning_rounds = 1;
    // knobs to control which schedules will be measured, refer to FLAGS_evaluate_knobs explanation
    std::bitset<3> evaluate_knobs = 7UL;
  };

  void SetUp() override { FLAGS_cinn_parallel_compile_size = 0; }

  void Evaluate(const frontend::Program& program) {
    if (FLAGS_evaluate_knobs >= 0) {
      options_.evaluate_knobs = FLAGS_evaluate_knobs;
    }
    VLOG(3) << "evaluate_knobs = " << options_.evaluate_knobs;

    auto worker_fn = [this, &program](const std::string& schedule_name, BuildRuntimeProgramFn build_fn) {
      VLOG(3) << "Initialize graph.";
      auto graph = std::make_shared<hlir::framework::Graph>(program, target_);
      VLOG(3) << "Apply graph pass.";
      hlir::framework::ApplyPass(graph.get(), "InferShape");
      hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
      VLOG(3) << "Build " << schedule_name << " program.";
      auto scope           = BuildScope(target_, graph);
      auto graph_compiler  = std::make_unique<GraphCompiler>(target_, scope, graph);
      auto runtime_program = (this->*build_fn)(graph.get(), graph_compiler.get());
      VLOG(3) << "Execute " << schedule_name << " program.";
      runtime_program->ExecuteTest(options_.repeat_times);
    };

    if (options_.evaluate_knobs.test(0)) {
      worker_fn("no schedule", &PerformanceTester::BuildNoScheduleProgram);
    }
    if (options_.evaluate_knobs.test(1)) {
      worker_fn("manual schedule", &PerformanceTester::BuildManualScheduleProgram);
    }
    if (options_.evaluate_knobs.test(2)) {
      worker_fn("auto schedule", &PerformanceTester::BuildAutoScheduleProgram);
    }
  }

 protected:
  using BuildRuntimeProgramFn = std::unique_ptr<hlir::framework::Program> (PerformanceTester::*)(Graph*,
                                                                                                 GraphCompiler*);

  std::unique_ptr<hlir::framework::Program> BuildNoScheduleProgram(Graph* graph, GraphCompiler* graph_compiler) {
    const auto& dtype_dict = graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>("inferdtype");
    const auto& shape_dict = graph->GetAttrs<absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");

    std::shared_ptr<hlir::framework::OpLowerer> op_lowerer =
        std::make_unique<hlir::framework::OpLowerer>(dtype_dict, shape_dict, target_);

    GraphCompiler::CompileOptions compile_options;
    compile_options.with_instantiate_variables = true;

    if (graph->fusion_groups.empty()) {
      compile_options.groups = hlir::pass::BuildNonFusedGroups(graph);
    } else {
      compile_options.groups = graph->fusion_groups;
    }

    for (auto group : graph->fusion_groups) {
      compile_options.lowered_funcs.push_back(op_lowerer->LowerWithoutSchedule(group));
    }

    VLOG(3) << "===========================No Schedule LoweredFunc Begin===========================";
    for (const auto& funcvec : compile_options.lowered_funcs) {
      for (const auto& func : funcvec) {
        VLOG(3) << func;
      }
    }
    VLOG(3) << "===========================No Schedule LoweredFunc End=============================";

    return graph_compiler->Build(compile_options).runtime_program;
  }

  std::unique_ptr<hlir::framework::Program> BuildManualScheduleProgram(Graph* graph, GraphCompiler* graph_compiler) {
    return graph_compiler->Build();
  }

  std::unique_ptr<hlir::framework::Program> BuildAutoScheduleProgram(Graph* graph, GraphCompiler* graph_compiler) {
    auto tuner = std::make_unique<AutoTuner>(target_, graph);

    AutoTuner::Config tuning_config;
    TuningOptions tuning_options;
    tuning_options.num_tuning_rounds         = options_.num_tuning_rounds;
    tuning_options.num_measure_trials        = 2;
    tuning_options.num_samples_per_iteration = 2;

    tuner->Initialize(tuning_config, graph_compiler);
    TuningResult tuning_result = tuner->Tune(tuning_options);

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

    return graph_compiler->Build(compile_options).runtime_program;
  }

#ifdef CINN_WITH_CUDA
  Target target_ = common::DefaultNVGPUTarget();
#else
  Target target_ = common::DefaultHostTarget();
#endif
  Options options_;
};

constexpr int batch_size = 2;

TEST_F(PerformanceTester, Mul) {
  int M = 32;
  int K = 16;
  int N = 32;

  Evaluate(MulProgramBuilder({M, K}, {N, K})());
}

TEST_F(PerformanceTester, Add) { Evaluate(AddProgramBuilder({1, 56, 56, 256}, {1, 56, 56, 256})()); }

TEST_F(PerformanceTester, Matmul) {
  int M = batch_size;
  int K = 2048;
  int N = 1000;

  Evaluate(MatmulProgramBuilder({M, K}, {K, N})());
}

TEST_F(PerformanceTester, Relu) { Evaluate(ReluProgramBuilder({batch_size, 64, 56, 56})()); }

TEST_F(PerformanceTester, Conv2d) {
  std::vector<int32_t> input_shape{batch_size, 3, 224, 224};
  std::vector<int32_t> weight_shape{64, 3, 7, 7};
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{3, 3};
  std::vector<int> dilations{1, 1};
  int groups                    = 1;
  std::string data_format       = "NCHW";
  std::string padding_algorithm = "EXPLICIT";

  Evaluate(Conv2dProgramBuilder(
      input_shape, weight_shape, strides, paddings, dilations, groups, data_format, padding_algorithm)());
}

TEST_F(PerformanceTester, Pool2d) {
  std::vector<int32_t> input_shape{batch_size, 64, 112, 112};
  std::string pooling_type = "max";
  std::vector<int> ksize{3, 3};
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{1, 1, 1, 1};
  bool ceil_mode                = false;
  bool exclusive                = true;
  bool global_pooling           = false;
  std::string data_format       = "NCHW";
  bool adaptive                 = false;
  std::string padding_algorithm = "EXPLICIT";

  Evaluate(Pool2dProgramBuilder(input_shape,
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

  Evaluate(BatchNormProgramBuilder(
      input_shape, scale_shape, bias_shape, mean_shape, variance_shape, epsilon, momentum, data_layout, is_test)());
}

TEST_F(PerformanceTester, Reshape) {
  std::vector<int32_t> input_shape{batch_size, 2048, 1, 1};
  std::vector<int32_t> output_shape{batch_size, 2048};

  Evaluate(ReshapeProgramBuilder(input_shape, output_shape)());
}

TEST_F(PerformanceTester, Softmax) {
  std::vector<int32_t> input_shape{batch_size, 1000};
  int axis                = -1;
  std::string data_format = "AnyLayout";

  Evaluate(SoftmaxProgramBuilder(input_shape, axis, data_format)());
}

TEST_F(PerformanceTester, Scale) {
  std::vector<int32_t> input_shape{batch_size, 1000};
  float scale           = 1.0f;
  float bias            = 0.0f;
  bool bias_after_scale = true;

  Evaluate(ScaleProgramBuilder(input_shape, scale, bias, bias_after_scale)());
}

// paddle model test
TEST_F(PerformanceTester, ResNet50) {
  std::vector<std::string> input_names       = {"inputs"};
  std::vector<std::vector<int>> input_shapes = {{batch_size, 3, 224, 224}};
  CHECK_NE(FLAGS_resnet50_model_dir, "");

  options_.evaluate_knobs = 0UL;
  Evaluate(PaddleModelProgramBuilder(FLAGS_resnet50_model_dir, input_names, input_shapes)());
}

}  // namespace auto_schedule
}  // namespace cinn

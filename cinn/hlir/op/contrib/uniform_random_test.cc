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
#include "cinn/hlir/op/contrib/uniform_random.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/codegen_c_x86.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/common/context.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/optimize.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/stage.h"
#include "cinn/utils/data_util.h"

namespace cinn {
namespace hlir {
namespace op {

#ifdef CINN_WITH_CUDA
TEST(GenerateCode_CUDA, UniformRandomGPU) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultNVGPUTarget();

  std::vector<int> shape = {128, 12};
  int seed               = 2023;
  std::string dtype      = "float32";

  ir::Tensor res = UniformRandom(shape, seed, dtype, target, "uniform_random_out");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeGPU_UniformRandom", stages, {res}, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before CUDA codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("UniformRandom_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  auto module                    = builder.Build();
  auto host_module_device_module = backends::SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module              = std::get<0>(host_module_device_module);
  auto& device_module            = std::get<1>(host_module_device_module);

  backends::CodeGenCUDA_Dev codegen(target);
  std::string source_code = codegen.Compile(device_module);
  LOG(INFO) << "compiled code:\n" << source_code;
}

}  // namespace op
}  // namespace hlir

namespace frontend {

TEST(Builder, UniformRandomFP32) {
  NetBuilder builder("net_builder");

  std::vector<int> shape = {128, 12, 128, 128};
  int seed               = 2023;
  std::string dtype      = "float32";
  auto out               = builder.UniformRandom(shape, 0., 1., seed, dtype);
  auto program           = builder.Build();

  for (int i = 0; i < program.size(); ++i) {
    LOG(INFO) << "instruction: " << program[i];
  }

  Target target = common::DefaultNVGPUTarget();
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  LOG(INFO) << "graph: \n" << graph->Visualize();

  auto scope = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  auto out_ten = scope->GetTensor(std::string(out->id));
  runtime_program->Execute();

  EXPECT_EQ(out_ten->type(), Float(32));

  std::vector<float> data = GetTensorData<float>(out_ten, target);

  int cnt = 0;
  for (int i = 0; i < 128 * 12 * 128 * 128; ++i) {
    if (data[i] > 0.5) cnt++;
  }
  float ratio = (float)cnt / (128 * 12 * 128 * 128);
  LOG(INFO) << "count: " << cnt;
  LOG(INFO) << "x > 0.5f ratio:  " << ratio;
  EXPECT_LE(ratio, 0.501f);
  EXPECT_GE(ratio, 0.499f);
}

TEST(Builder, UniformRandomFP64) {
  NetBuilder builder("net_builder");

  std::vector<int> shape = {128, 12, 128, 128};
  int seed               = 2023;
  std::string dtype      = "float64";
  auto out               = builder.UniformRandom(shape, 0., 1., seed, dtype);
  auto program           = builder.Build();

  for (int i = 0; i < program.size(); ++i) {
    LOG(INFO) << "instruction: " << program[i];
  }

  Target target = common::DefaultNVGPUTarget();
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  LOG(INFO) << "graph: \n" << graph->Visualize();

  auto scope = BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  auto out_ten = scope->GetTensor(std::string(out->id));
  runtime_program->Execute();

  EXPECT_EQ(out_ten->type(), Float(64));

  std::vector<double> data = GetTensorData<double>(out_ten, target);

  int cnt = 0;
  for (int i = 0; i < 128 * 12 * 128 * 128; ++i) {
    if (data[i] > 0.5) cnt++;
  }

  float ratio = (float)cnt / (128 * 12 * 128 * 128);
  LOG(INFO) << "count: " << cnt;
  LOG(INFO) << "x > 0.5f ratio:  " << ratio;
  EXPECT_LE(ratio, 0.501f);
  EXPECT_GE(ratio, 0.499f);
}
#endif

}  // namespace frontend

}  // namespace cinn

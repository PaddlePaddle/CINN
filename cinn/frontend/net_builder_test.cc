// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/frontend/net_builder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace frontend {

using hlir::framework::OpRegistry;

namespace {
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

void SetRandData(hlir::framework::Tensor tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = dist(engine);  // All random data
  }

#ifdef CINN_WITH_CUDA
  cudaMemcpy(data, random_data.data(), num_ele * sizeof(float), cudaMemcpyHostToDevice);
#else
  std::copy(random_data.begin(), random_data.end(), data);
#endif
}

template <typename T, typename Alloc = std::allocator<T>>
std::ostream& operator<<(std::ostream& os, const std::vector<T, Alloc>& vec) {
  os << "{ ";
  for (auto e : vec) {
    os << e << " ";
  }
  os << "}\n";
  return os;
}
}  // namespace

TEST(net_build, basic) {
  LOG(INFO) << "The size of registered operators: " << OpRegistry::Global()->ListAllNames().size();
  LOG(INFO) << "Registered operators:\n" << OpRegistry::Global()->ListAllNames();
  auto program = CreateAddProgram();
  // output program
  for (int i = 0; i < program.size(); i++) {
    LOG(INFO) << "instruction: " << program[i];
  }
}

TEST(net_build, program_execute_multi_elementwise_add) {
  auto program = CreateAddProgram();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A = scope->GetTensor("A");
  auto B = scope->GetTensor("B");
  SetRandData(A, target);
  SetRandData(B, target);

  runtime_program->Execute();
}

TEST(net_build, program_execute_fc) {
  constexpr int B = 10;  // batch size
  constexpr int M = 32;
  constexpr int K = 18;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {B, M, K}, "A");
  auto w = builder.CreateInput(Float(32), {N, K}, "W");  // weight
  auto b = builder.CreateInput(Float(32), {N}, "B");     // bias

  auto mul_out = builder.Mul(a, w, 2, 1);
  auto add_out = builder.Add(mul_out, b);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(a.id()));
  scope->Var<hlir::framework::Tensor>(std::string(w.id()));
  scope->Var<hlir::framework::Tensor>(std::string(b.id()));
  scope->Var<hlir::framework::Tensor>(std::string(mul_out->id));

  auto a_ten        = scope->GetTensor(std::string(a.id()));
  auto w_ten        = scope->GetTensor(std::string(w.id()));
  auto b_ten        = scope->GetTensor(std::string(b.id()));
  auto fake_out_ten = scope->GetTensor(std::string(mul_out->id));
  auto add_out_ten  = scope->GetTensor(std::string(add_out->id));
  SetRandData(a_ten, target);
  SetRandData(w_ten, target);
  SetRandData(b_ten, target);

  runtime_program->Execute();
}

TEST(net_build, program_execute_reverse) {
  const int B = 16;
  const int C = 3;
  const int H = 224;
  const int W = 224;

  NetBuilder builder("net_builder");
  Placeholder input    = builder.CreateInput(Float(32), {B, C, H, W}, "Img");
  Variable reverse_out = builder.Reverse(input, {2, 3});
  auto program         = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(reverse_out->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData(input_tensor, target);
  runtime_program->Execute();
}

void SetIntRandData(hlir::framework::Tensor tensor, Target target, int min = 0, int max = 127) {
  auto* data = tensor->mutable_data<int>(target);
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_int_distribution<int> dist(min, max);
  size_t num_ele = tensor->shape().numel();
  std::vector<int> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = dist(engine);  // All random data
  }
  std::copy(random_data.begin(), random_data.end(), data);
}

void SetFloatRandData(hlir::framework::Tensor tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = dist(engine);  // All random data
  }
  std::copy(random_data.begin(), random_data.end(), data);
}

TEST(net_build, program_execute_gather) {
  const int B     = 4;
  const int H_IN1 = 7;
  const int H_IN2 = 14;

  NetBuilder builder("net_builder");
  Placeholder input1 = builder.CreateInput(Float(32), {B, H_IN1}, "In1");
  Placeholder input2 = builder.CreateInput(Int(32), {B, H_IN2}, "In2");
  Variable output    = builder.Gather(input1, input2, 1);
  auto program       = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input1.id()));
  scope->Var<hlir::framework::Tensor>(std::string(input2.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input1_tensor = scope->GetTensor(std::string(input1.id()));
  SetFloatRandData(input1_tensor, target);
  float* input1_data = input1_tensor->mutable_data<float>(target);

  auto input2_tensor = scope->GetTensor(std::string(input2.id()));
  SetIntRandData(input2_tensor, target, 0, 6);
  int* input2_data = input2_tensor->mutable_data<int>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H_IN2);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H_IN2; ++h) {
      std::string line;
      int index      = h + H_IN2 * b;
      float in_data  = input1_data[input2_data[index] + H_IN1 * b];
      float out_data = output_data[index];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(in_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_gather_nd) {
  const int B     = 4;
  const int H_IN1 = 7;
  const int H_IN2 = 14;

  NetBuilder builder("net_builder");
  Placeholder input1 = builder.CreateInput(Float(32), {B, H_IN1}, "In1");
  Placeholder input2 = builder.CreateInput(Int(32), {B, H_IN2, 1}, "In2");
  Variable output    = builder.GatherNd(input1, input2, {1});
  auto program       = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input1.id()));
  scope->Var<hlir::framework::Tensor>(std::string(input2.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input1_tensor = scope->GetTensor(std::string(input1.id()));
  SetFloatRandData(input1_tensor, target);
  float* input1_data = input1_tensor->mutable_data<float>(target);

  auto input2_tensor = scope->GetTensor(std::string(input2.id()));
  SetIntRandData(input2_tensor, target, 0, 6);
  int* input2_data = input2_tensor->mutable_data<int>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H_IN2);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H_IN2; ++h) {
      std::string line;
      int index      = h + H_IN2 * b;
      float in_data  = input1_data[input2_data[index] + H_IN1 * b];
      float out_data = output_data[index];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(in_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_scatter) {
  const int B     = 3;
  const int H_IN  = 4;
  const int H_OUT = 7;

  NetBuilder builder("net_builder");
  Placeholder input1 = builder.CreateInput(Float(32), {B, H_IN}, "In1");
  Placeholder input2 = builder.CreateInput(Int(32), {B, H_IN}, "In2");
  Variable output    = builder.Scatter(input1, input2, {B, H_OUT}, 0, 1);
  auto program       = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input1.id()));
  scope->Var<hlir::framework::Tensor>(std::string(input2.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input1_tensor = scope->GetTensor(std::string(input1.id()));
  SetFloatRandData(input1_tensor, target);
  float* input1_data = input1_tensor->mutable_data<float>(target);

  auto input2_tensor = scope->GetTensor(std::string(input2.id()));
  SetIntRandData(input2_tensor, target, 0, 6);
  int* input2_data = input2_tensor->mutable_data<int>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H_OUT);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H_IN; ++h) {
      std::string line;
      int index      = h + H_IN * b;
      float in_data  = input1_data[index];
      float out_data = output_data[input2_data[index] + H_OUT * b];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(in_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_scatter_nd) {
  const int B     = 3;
  const int H_IN  = 4;
  const int H_OUT = 7;

  NetBuilder builder("net_builder");
  Placeholder input1 = builder.CreateInput(Float(32), {B, H_IN}, "In1");
  Placeholder input2 = builder.CreateInput(Int(32), {B, H_IN, 1}, "In2");
  Variable output    = builder.ScatterNd(input1, input2, {B, H_OUT}, 0, {1});
  auto program       = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input1.id()));
  scope->Var<hlir::framework::Tensor>(std::string(input2.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input1_tensor = scope->GetTensor(std::string(input1.id()));
  SetFloatRandData(input1_tensor, target);
  float* input1_data = input1_tensor->mutable_data<float>(target);

  auto input2_tensor = scope->GetTensor(std::string(input2.id()));
  SetIntRandData(input2_tensor, target, 0, 6);
  int* input2_data = input2_tensor->mutable_data<int>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H_OUT);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H_IN; ++h) {
      std::string line;
      int index      = h + H_IN * b;
      float in_data  = input1_data[index];
      float out_data = output_data[input2_data[index] + H_OUT * b];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(in_data, out_data);
      VLOG(6) << line;
    }
  }
}

}  // namespace frontend
}  // namespace cinn

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

void SetRandCPUData(hlir::framework::Tensor tensor, Target target) {
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

TEST(net_build, program_argmax_case1) {
  const int N     = 4;
  const int IN_C  = 3;
  const int OUT_C = 1;
  const int H     = 7;
  const int W     = 7;

  NetBuilder builder("net_builder");
  Placeholder input  = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Placeholder output = builder.Argmax(input, 0, true);
  auto program       = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetSqueezeRandData(input_tensor, target);
  float* input_data = input_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize input_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          line += (std::to_string(input_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 4UL);
  EXPECT_EQ(output_shape[0], N);
  EXPECT_EQ(output_shape[1], OUT_C);
  EXPECT_EQ(output_shape[2], H);
  EXPECT_EQ(output_shape[3], W);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < OUT_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index       = w + W * (h + H * (c + OUT_C * n));
          float in_data   = input_data[index];
          int out_data    = output_data[index];
          int max_index   = w + W * (h + H * (out_data + OUT_C * n));
          float max_value = input_data[max_index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_LE(in_data, max_value);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_argmax_case2) {
  const int N    = 4;
  const int IN_C = 3;
  const int H    = 7;
  const int W    = 7;

  NetBuilder builder("net_builder");
  Placeholder input  = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Placeholder output = builder.Argmax(input, 0, false);
  auto program       = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetSqueezeRandData(input_tensor, target);
  float* input_data = input_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize input_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          line += (std::to_string(input_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], N);
  EXPECT_EQ(output_shape[1], H);
  EXPECT_EQ(output_shape[2], W);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    VLOG(6) << "n = " << n << ", c = " << c;
    for (int h = 0; h < H; ++h) {
      std::string line;
      for (int w = 0; w < W; ++w) {
        int index       = w + W * (h + H * n);
        float in_data   = input_data[index];
        int out_data    = output_data[index];
        int max_index   = w + W * (h + H * (out_data + n));
        float max_value = input_data[max_index];
        line += (std::to_string(out_data) + ", ");
        EXPECT_LE(in_data, max_value);
      }
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_argmin_case1) {
  const int N     = 4;
  const int IN_C  = 3;
  const int OUT_C = 1;
  const int H     = 7;
  const int W     = 7;

  NetBuilder builder("net_builder");
  Placeholder input  = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Placeholder output = builder.Argmin(input, 0, true);
  auto program       = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetSqueezeRandData(input_tensor, target);
  float* input_data = input_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize input_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          line += (std::to_string(input_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 4UL);
  EXPECT_EQ(output_shape[0], N);
  EXPECT_EQ(output_shape[1], OUT_C);
  EXPECT_EQ(output_shape[2], H);
  EXPECT_EQ(output_shape[3], W);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < OUT_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index       = w + W * (h + H * (c + OUT_C * n));
          float in_data   = input_data[index];
          int out_data    = output_data[index];
          int max_index   = w + W * (h + H * (out_data + OUT_C * n));
          float max_value = input_data[max_index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_GE(in_data, max_value);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_argmin_case2) {
  const int N    = 4;
  const int IN_C = 3;
  const int H    = 7;
  const int W    = 7;

  NetBuilder builder("net_builder");
  Placeholder input  = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Placeholder output = builder.Argmin(input, 0, false);
  auto program       = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetSqueezeRandData(input_tensor, target);
  float* input_data = input_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize input_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          line += (std::to_string(input_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], N);
  EXPECT_EQ(output_shape[1], H);
  EXPECT_EQ(output_shape[2], W);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    VLOG(6) << "n = " << n << ", c = " << c;
    for (int h = 0; h < H; ++h) {
      std::string line;
      for (int w = 0; w < W; ++w) {
        int index       = w + W * (h + H * n);
        float in_data   = input_data[index];
        int out_data    = output_data[index];
        int max_index   = w + W * (h + H * (out_data + n));
        float max_value = input_data[max_index];
        line += (std::to_string(out_data) + ", ");
        EXPECT_GE(in_data, max_value);
      }
      VLOG(6) << line;
    }
  }
}


}  // namespace frontend
}  // namespace cinn

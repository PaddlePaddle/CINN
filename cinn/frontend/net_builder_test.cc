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
#include "cinn/utils/data_util.h"
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
  SetRandData<float>(A, target);
  SetRandData<float>(B, target);

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
  SetRandData<float>(a_ten, target);
  SetRandData<float>(w_ten, target);
  SetRandData<float>(b_ten, target);

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
  SetRandData<float>(input_tensor, target);
  runtime_program->Execute();
}

TEST(net_build, program_execute_clip) {
  const int M = 4;
  const int N = 3;
  const int K = 7;

  const float max_val = 0.8;
  const float min_val = 0.2;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {M, N, K}, "In");
  Variable output   = builder.Clip({input}, max_val, min_val);
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  float* input_data = input_tensor->mutable_data<float>(target);

  memset(input_data, 0, sizeof(float) * M * N * K);

  VLOG(6) << "Visualize input_data";

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      VLOG(6) << "m = " << m << ", n = " << n;
      std::string line;
      for (int k = 0; k < K; ++k) {
        int index         = m * (N * K) + n * K + k;
        input_data[index] = rand() % 1000 / 1000.f;
        line += (std::to_string(input_data[index]) + ", ");
      }
      VLOG(6) << line;
    }
  }

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], M);
  EXPECT_EQ(output_shape[1], N);
  EXPECT_EQ(output_shape[2], K);

  float* output_data = output_tensor->mutable_data<float>(target);

  VLOG(6) << "Visualize output_data";

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      VLOG(6) << "m = " << m << ", n = " << n;
      std::string line;
      for (int k = 0; k < K; ++k) {
        int index      = m * (N * K) + n * K + k;
        float in_data  = input_data[index];
        float out_data = output_data[index];
        in_data        = in_data < min_val ? min_val : in_data;
        in_data        = in_data > max_val ? max_val : in_data;
        EXPECT_EQ(in_data, out_data);
        line += (std::to_string(out_data) + ", ");
      }
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_cast) {
  const int B = 4;
  const int H = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Int(32), {B, H}, "In");
  Variable output   = builder.Cast(input, "float");
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<int>(input_tensor, target);
  int* input_data = input_tensor->mutable_data<int>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      std::string line;
      int index      = h + H * b;
      float in_data  = (float)input_data[index];
      float out_data = output_data[index];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(in_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_squeeze_case1) {
  const int B = 4;
  const int C = 1;
  const int H = 7;
  const int W = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "In");
  Variable output   = builder.Squeeze(input, {1});
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  float* input_data = input_tensor->mutable_data<float>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);
  EXPECT_EQ(output_shape[2], W);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index      = w + W * (h + H * (c + C * b));
          float in_data  = input_data[index];
          float out_data = output_data[index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_EQ(in_data, out_data);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_squeeze_case2) {
  const int B = 4;
  const int C = 1;
  const int H = 7;
  const int W = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "In");
  Variable output   = builder.Squeeze(input, {1, 3});
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  float* input_data = input_tensor->mutable_data<float>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index      = w + W * (h + H * (c + C * b));
          float in_data  = input_data[index];
          float out_data = output_data[index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_EQ(in_data, out_data);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_squeeze_case3) {
  const int B = 4;
  const int C = 1;
  const int H = 7;
  const int W = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "In");
  Variable output   = builder.Squeeze(input, {});
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  float* input_data = input_tensor->mutable_data<float>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index      = w + W * (h + H * (c + C * b));
          float in_data  = input_data[index];
          float out_data = output_data[index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_EQ(in_data, out_data);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_argsort) {
  const int B = 4;
  const int H = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, H}, "In");
  Variable output   = builder.ArgSort(input, 0, true);
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  auto* input_data = input_tensor->mutable_data<float>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Int(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  int* output_data = output_tensor->mutable_data<int>(target);
  VLOG(6) << "Visualize output_data";
  for (int h = 0; h < H; ++h) {
    std::vector<float> sorted_data;
    std::vector<float> out_sorted_data(H);
    for (int b = 0; b < B; ++b) {
      int index = h + H * b;
      sorted_data.push_back(input_data[index]);
      out_sorted_data[b] = input_data[h + H * output_data[index]];
    }
    std::sort(sorted_data.begin(), sorted_data.begin() + B);

    for (int b = 0; b < B; ++b) {
      std::string line;
      int index       = h + H * b;
      float true_data = sorted_data[b];
      float out_data  = out_sorted_data[b];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(true_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_sort) {
  const int B = 4;
  const int H = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, H}, "In");
  Variable output   = builder.Sort(input, 0, true);
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  auto* input_data = input_tensor->mutable_data<float>(target);

  runtime_program->Execute();

  auto output_tensor                   = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  float* output_data = output_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize output_data";
  for (int h = 0; h < H; ++h) {
    std::vector<float> sorted_data;
    for (int b = 0; b < B; ++b) {
      int index = h + H * b;
      sorted_data.push_back(input_data[index]);
    }
    std::sort(sorted_data.begin(), sorted_data.begin() + B);

    for (int b = 0; b < B; ++b) {
      std::string line;
      int index       = h + H * b;
      float true_data = sorted_data[b];
      float out_data  = output_data[index];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(true_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_arange_float) {
  const float start       = 1.5F;
  const float stop        = 31.5F;
  const float step        = 2.0F;
  const std::string dtype = "float32";

  NetBuilder builder("net_builder");
  Variable out = builder.Arange(start, stop, step, dtype);
  auto program = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(out->id));

  runtime_program->Execute();

  auto out_tensor                          = scope->GetTensor(std::string(out->id));
  const std::vector<int>& out_tensor_shape = out_tensor->shape().data();
  EXPECT_EQ(out_tensor->type(), Float(32));
  EXPECT_EQ(out_tensor_shape.size(), 1UL);

  int num_elem = static_cast<int>(std::ceil((stop - start) / step));
  EXPECT_EQ(out_tensor_shape[0], num_elem);

  float* out_data = out_tensor->mutable_data<float>(target);
  for (int i = 0; i < out_tensor_shape[0]; ++i) {
    EXPECT_NEAR(out_data[i], start + step * i, 1e-5);
    VLOG(6) << out_data[i];
  }
}

TEST(net_build, program_execute_arange_int) {
  const float start       = 1.5F;
  const float stop        = 31.5F;
  const float step        = 1.6F;
  const std::string dtype = "int32";

  NetBuilder builder("net_builder");
  Variable out = builder.Arange(start, stop, step, dtype);
  auto program = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(out->id));

  runtime_program->Execute();

  auto out_tensor                          = scope->GetTensor(std::string(out->id));
  const std::vector<int>& out_tensor_shape = out_tensor->shape().data();
  EXPECT_EQ(out_tensor->type(), Int(32));
  EXPECT_EQ(out_tensor_shape.size(), 1UL);

  int num_elem = static_cast<int>(std::ceil((stop - start) / step));
  EXPECT_EQ(out_tensor_shape[0], num_elem);

  int32_t* out_data = out_tensor->mutable_data<int32_t>(target);
  for (int i = 0; i < out_tensor_shape[0]; ++i) {
    EXPECT_EQ(out_data[i], static_cast<int32_t>(start + step * i));
    VLOG(6) << out_data[i];
  }
}

TEST(net_build, program_argmax_case1) {
  const int N     = 4;
  const int IN_C  = 3;
  const int OUT_C = 1;
  const int H     = 7;
  const int W     = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Variable output   = builder.Argmax(input, 1, true);
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
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

  int* output_data = output_tensor->mutable_data<int>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index     = w + W * (h + H * (c + IN_C * n));
          float in_data = input_data[index];
          int out_index = w + W * (h + H * n);
          int out_data  = output_data[out_index];
          EXPECT_LE(0, out_data);
          EXPECT_LT(out_data, IN_C);
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
  Placeholder input = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Variable output   = builder.Argmax(input, 1, false);
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
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

  int* output_data = output_tensor->mutable_data<int>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index     = w + W * (h + H * (c + IN_C * n));
          float in_data = input_data[index];
          int out_index = w + W * (h + H * n);
          int out_data  = output_data[index];
          EXPECT_LE(0, out_data);
          EXPECT_LT(out_data, IN_C);
          int max_index   = w + W * (h + H * (out_data + n));
          float max_value = input_data[max_index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_LE(in_data, max_value);
        }
        VLOG(6) << line;
      }
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
  Placeholder input = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Variable output   = builder.Argmin(input, 1, true);
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
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

  int* output_data = output_tensor->mutable_data<int>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index     = w + W * (h + H * (c + IN_C * n));
          float in_data = input_data[index];
          int out_index = w + W * (h + H * n);
          int out_data  = output_data[index];

          EXPECT_LE(0, out_data);
          EXPECT_LT(out_data, IN_C);
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
  Placeholder input = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Variable output   = builder.Argmin(input, 1, false);
  auto program      = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
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

  int* output_data = output_tensor->mutable_data<int>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index     = w + W * (h + H * (c + IN_C * n));
          float in_data = input_data[index];
          int out_index = w + W * (h + H * n);
          int out_data  = output_data[index];
          EXPECT_LE(0, out_data);
          EXPECT_LT(out_data, IN_C);
          int max_index   = w + W * (h + H * (out_data + n));
          float max_value = input_data[max_index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_GE(in_data, max_value);
        }
        VLOG(6) << line;
      }
    }
  }
}

}  // namespace frontend
}  // namespace cinn

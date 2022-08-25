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
#include <cstring>
#include <memory>
#include <random>
#include <string>
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

TEST(net_build, program_execute_pool2d_grad) {
  const int B        = 4;
  const int C        = 3;
  const int IN_H     = 7;
  const int IN_W     = 7;
  const int OUT_H    = 3;
  const int OUT_W    = 3;
  const int KERNEL_H = 5;
  const int KERNEL_W = 5;

  NetBuilder builder("net_builder");
  Placeholder input    = builder.CreateInput(Float(32), {B, C, IN_H, IN_W}, "In");
  Placeholder output   = builder.CreateInput(Float(32), {B, C, OUT_H, OUT_W}, "Out");
  Placeholder out_grad = builder.CreateInput(Float(32), {B, C, OUT_H, OUT_W}, "OutGrad");
  Variable in_grad     = builder.Pool2dGrad(input, output, out_grad, {KERNEL_H, KERNEL_W});
  auto program         = builder.Build();

  Target target = common::DefaultHostTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output.id()));
  scope->Var<hlir::framework::Tensor>(std::string(out_grad.id()));
  scope->Var<hlir::framework::Tensor>(std::string(in_grad->id));

  auto out_grad_tensor = scope->GetTensor(std::string(out_grad.id()));
  float* out_grad_data = out_grad_tensor->mutable_data<float>(target);
  memset(out_grad_data, 0, sizeof(float) * B * C * OUT_H * OUT_W);
  out_grad_data[0] = static_cast<float>(KERNEL_H * KERNEL_W);
  out_grad_data[1] = static_cast<float>(KERNEL_H * KERNEL_W);

  VLOG(6) << "Visualize out_grad_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < OUT_H; ++h) {
        std::string line;
        for (int w = 0; w < OUT_W; ++w) {
          int index = w + OUT_W * (h + OUT_H * (c + C * b));
          line += (std::to_string(out_grad_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto in_grad_tensor                   = scope->GetTensor(std::string(in_grad->id));
  const std::vector<int>& in_grad_shape = in_grad_tensor->shape().data();
  EXPECT_EQ(in_grad_shape.size(), 4UL);
  EXPECT_EQ(in_grad_shape[0], B);
  EXPECT_EQ(in_grad_shape[1], C);
  EXPECT_EQ(in_grad_shape[2], IN_H);
  EXPECT_EQ(in_grad_shape[3], IN_W);

  float* in_grad_data = in_grad_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize in_grad_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < IN_H; ++h) {
        std::string line;
        for (int w = 0; w < IN_W; ++w) {
          int index  = w + IN_W * (h + IN_H * (c + C * b));
          float data = in_grad_data[index];
          line += (std::to_string(data) + ", ");
          if (b == 0 && c == 0 && h < KERNEL_H) {
            if (w == 0 || w == KERNEL_W) {
              EXPECT_EQ(data, 1.0f);
            } else if (w > 0 && w < KERNEL_W) {
              EXPECT_EQ(data, 2.0f);
            } else {
              EXPECT_EQ(data, 0.0f);
            }
          }
        }
        VLOG(6) << line;
      }
    }
  }
}

}  // namespace frontend
}  // namespace cinn

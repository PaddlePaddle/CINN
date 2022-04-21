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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <unordered_set>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"
#include "gtest/gtest.h"

namespace cinn::frontend {

namespace {

bool IsCompiledWithCUDA() {
#if !defined(CINN_WITH_CUDA)
  return false;
#else
  return true;
#endif
}

void PrintMatrix(const std::vector<float>& mat, int bs, int m, int n) {
  if (!VLOG_IS_ON(4)) {
    return;
  }
  const auto min_max = std::minmax_element(mat.begin(), mat.end());
  int min            = static_cast<int>(*min_max.first);
  int max            = static_cast<int>(*min_max.second);
  auto ele_width     = std::max(std::to_string(min).length(), std::to_string(max).length());
  std::cout << "\n" << std::string((ele_width + 2) * n - 1, '-') << "\n";
  for (int b = 0; b < bs; b++) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        std::cout << std::setw(ele_width) << mat[b * m * n + i * n + j] << ", ";
      }
      std::cout << "\n";
    }
    if (b != bs - 1) {
      std::cout << std::string((ele_width + 2) * n - 1, '*') << "\n";
    }
  }
  std::cout << std::string((ele_width + 2) * n - 1, '-') << "\n\n";
}

void SetRandData(hlir::framework::Tensor tensor, const Target& target, int seed = -1) {
  if (seed == -1) {
    std::random_device rd;
    seed = rd();
  }
  std::default_random_engine engine(seed);
  std::uniform_int_distribution<int> dist(1, 10);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = static_cast<float>(dist(engine));  // All random data
  }

  auto* data = tensor->mutable_data<float>(target);
#ifdef CINN_WITH_CUDA
  cudaMemcpy(data, random_data.data(), num_ele * sizeof(float), cudaMemcpyHostToDevice);
#else
  std::copy(random_data.begin(), random_data.end(), data);
#endif
}

std::vector<float> GetTensorData(const hlir::framework::Tensor& tensor, Target target) {
  auto size = tensor->shape().numel();
  std::vector<float> data(size);
#ifdef CINN_WITH_CUDA
  cudaMemcpy(
      data.data(), static_cast<const void*>(tensor->data<float>()), size * sizeof(float), cudaMemcpyDeviceToHost);
#else
  std::copy(tensor->data<float>(), tensor->data<float>() + size, data.begin());
#endif
  return data;
}

void RunGraph(std::shared_ptr<hlir::framework::Graph> graph,
              const Target& target,
              const std::shared_ptr<hlir::framework::Scope>& scope) {
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  VLOG(4) << "Graph Viz:\n" << graph->Visualize();
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

std::vector<float> RunProgram(const Program& program,
                              const Target& target,
                              const std::vector<std::string>& input_ids,
                              const std::vector<std::string>& output_ids,
                              int seed          = -1,
                              bool print_tensor = false) {
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = hlir::framework::BuildScope(target, graph);
  for (auto& input_id : input_ids) {
    scope->Var<hlir::framework::Tensor>(input_id);
    auto input_tensor = scope->GetTensor(input_id);
    SetRandData(input_tensor, target, seed);
    if (print_tensor) {
      auto tensor_data = GetTensorData(input_tensor, target);
      if (input_tensor->shape().data().size() == 2) {
        PrintMatrix(tensor_data, 1, input_tensor->shape().data()[0], input_tensor->shape().data()[1]);
      } else if (input_tensor->shape().data().size() == 3) {
        PrintMatrix(tensor_data,
                    input_tensor->shape().data()[0],
                    input_tensor->shape().data()[1],
                    input_tensor->shape().data()[2]);
      }
    }
  }

  RunGraph(graph, target, scope);

  auto output_tensor = scope->GetTensor(output_ids.front());
  auto output_data   = GetTensorData(output_tensor, target);
  if (print_tensor) {
    if (output_tensor->shape().data().size() == 2) {
      PrintMatrix(output_data, 1, output_tensor->shape().data()[0], output_tensor->shape().data()[1]);
    } else if (output_tensor->shape().data().size() == 3) {
      PrintMatrix(output_data,
                  output_tensor->shape().data()[0],
                  output_tensor->shape().data()[1],
                  output_tensor->shape().data()[2]);
    }
  }
  return output_data;
}

void CompareResult(Program* program,
                   const Target& target,
                   const std::vector<std::string>& input_ids,
                   const std::vector<std::string>& output_ids,
                   size_t size_diff,
                   int seed          = -1,
                   bool print_tensor = false) {
  std::unordered_set<std::string> fetch_ids(output_ids.begin(), output_ids.end());
  // apply common pass
  ProgramPass::Apply(program, fetch_ids, target, {"Decomposer", "RemoveIdentity"});

  // get original program size
  auto origin_size = program->size();
  // get original output
  auto origin_out = RunProgram(*program, target, input_ids, output_ids, seed, print_tensor);

  // fuse transpose + add + dot, then run and get the fused output
  ProgramPass::Apply(program, fetch_ids, target, {"TransposeFolding", "GemmRewriter"});

  // get fused program size
  auto fused_size = program->size();
  ASSERT_EQ(size_diff, origin_size - fused_size);
  // get fused output
  auto fused_out = RunProgram(*program, target, input_ids, output_ids, seed, print_tensor);

  ASSERT_EQ(origin_out.size(), fused_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], fused_out[i]);
  }
}

}  // namespace

TEST(GemmRwriter, BatchedTransLeft) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {3, 6, 8}, "A");
  auto b       = builder.Transpose(a, {0, 2, 1});
  auto c       = builder.CreateInput(Float(32), {3, 6, 7}, "C");
  auto d       = builder.Matmul(b, c);
  auto e       = builder.CreateInput(Float(32), {3, 8, 7}, "E");
  auto out     = builder.Add(d, e);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), e.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 2, 123, true);
}

TEST(GemmRwriter, BatchedTransRight) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {3, 8, 6}, "A");
  auto b       = builder.CreateInput(Float(32), {3, 7, 6}, "B");
  auto c       = builder.Transpose(b, {0, 2, 1});
  auto e       = builder.Matmul(a, c);
  auto f       = builder.CreateInput(Float(32), {3, 8, 7}, "F");
  auto out     = builder.Add(e, f);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 2, 123, true);
}

TEST(GemmRwriter, BatchedTransTwo) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {3, 6, 8}, "A");
  auto b       = builder.Transpose(a, {0, 2, 1});
  auto c       = builder.CreateInput(Float(32), {3, 7, 6}, "C");
  auto d       = builder.Transpose(c, {0, 2, 1});
  auto e       = builder.Matmul(b, d);
  auto f       = builder.CreateInput(Float(32), {3, 8, 7}, "F");
  auto out     = builder.Add(e, f);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 3, 123, true);
}

TEST(GemmRwriter, BatchedNoTrans) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {3, 8, 6}, "A");
  auto b       = builder.CreateInput(Float(32), {3, 6, 7}, "B");
  auto e       = builder.Matmul(a, b);
  auto f       = builder.CreateInput(Float(32), {3, 8, 7}, "F");
  auto out     = builder.Add(e, f);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 1, 123, true);
}

TEST(GemmRwriter, TransLeft) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b       = builder.Transpose(a, {1, 0});
  auto c       = builder.CreateInput(Float(32), {6, 7}, "C");
  auto d       = builder.Matmul(b, c);
  auto e       = builder.CreateInput(Float(32), {8, 7}, "E");
  auto out     = builder.Add(d, e);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), e.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 2, 123, true);
}

TEST(GemmRwriter, TransRight) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {8, 6}, "A");
  auto b       = builder.CreateInput(Float(32), {7, 6}, "B");
  auto c       = builder.Transpose(b, {1, 0});
  auto e       = builder.Matmul(a, c);
  auto f       = builder.CreateInput(Float(32), {8, 7}, "F");
  auto out     = builder.Add(e, f);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 2, 123, true);
}

TEST(GemmRwriter, TransTwo) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b       = builder.Transpose(a, {1, 0});
  auto c       = builder.CreateInput(Float(32), {7, 6}, "C");
  auto d       = builder.Transpose(c, {1, 0});
  auto e       = builder.Matmul(b, d);
  auto f       = builder.CreateInput(Float(32), {8, 7}, "F");
  auto out     = builder.Add(e, f);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), c.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 3, 123, true);
}

TEST(GemmRwriter, NoTrans) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {8, 6}, "A");
  auto b       = builder.CreateInput(Float(32), {6, 7}, "B");
  auto e       = builder.Matmul(a, b);
  auto f       = builder.CreateInput(Float(32), {8, 7}, "F");
  auto out     = builder.Add(e, f);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 1, 123, true);
}

TEST(GemmRwriter, BatchedComplex) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.FillConstant<float>({2, 20}, 2.0f, "A");
  auto b       = builder.BroadcastTo(a, {16, 2, 20}, {1, 2});
  auto c       = builder.Transpose(b, {0, 2, 1});
  auto d       = builder.CreateInput(Float(32), {121, 20}, "C");
  auto e       = builder.BroadcastTo(d, {16, 121, 20}, {1, 2});
  auto f       = builder.Matmul(e, c);
  auto x       = builder.FillConstant<float>({16, 2, 20}, 1.0f, "X");
  auto y       = builder.Transpose(x, {0, 2, 1});
  auto z       = builder.CreateInput(Float(32), {16, 20, 121}, "Z");
  auto l       = builder.Transpose(z, {0, 2, 1});
  auto m       = builder.Matmul(l, y);
  auto n       = builder.Mul(d, a);
  auto o       = builder.BroadcastTo(n, {16, n->shape[0], n->shape[1]}, {1, 2});
  auto p       = builder.Sub(f, o);
  auto q       = builder.Add(f, m);
  auto out     = builder.Add(p, q);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{d.id(), z.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 4, 123, false);
}

TEST(GemmRwriter, Complex) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.FillConstant<float>({2, 20}, 2.0f, "A");
  auto b       = builder.Transpose(a, {1, 0});
  auto c       = builder.CreateInput(Float(32), {121, 20}, "C");
  auto d       = builder.Matmul(c, b);
  auto x       = builder.FillConstant<float>({2, 20}, 1.0f, "X");
  auto y       = builder.Transpose(x, {1, 0});
  auto z       = builder.CreateInput(Float(32), {20, 121}, "Z");
  auto l       = builder.Transpose(z, {1, 0});
  auto m       = builder.Matmul(l, y);
  auto n       = builder.Mul(c, a);
  auto p       = builder.Sub(d, n);
  auto q       = builder.Add(d, m);
  auto out     = builder.Add(p, q);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{c.id(), z.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {out->id}, 4, 123, false);
}

}  // namespace cinn::frontend

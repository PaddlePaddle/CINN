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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "cinn/common/target.h"
#include "cinn/frontend/optimize.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

DECLARE_bool(cinn_use_new_fusion_pass);

namespace cinn::frontend {

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

void SetRandData(hlir::framework::Tensor tensor, const common::Target& target, int seed = -1) {
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

std::vector<float> GetTensorData(const hlir::framework::Tensor& tensor, const common::Target& target) {
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
              const common::Target& target,
              const std::shared_ptr<hlir::framework::Scope>& scope) {
  if (FLAGS_cinn_use_new_fusion_pass) {
    hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});
  } else {
    hlir::framework::ApplyPass(graph.get(), "OpFusion");
  }
  VLOG(4) << "Graph Viz:\n" << graph->Visualize();
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

std::vector<float> RunProgram(const Program& program,
                              const common::Target& target,
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
                   const common::Target& target,
                   const std::vector<std::string>& input_ids,
                   const std::vector<std::string>& output_ids,
                   size_t size_diff,
                   const std::pair<std::vector<std::string>, std::vector<std::string>>& passes,
                   int seed          = -1,
                   bool print_tensor = false) {
  std::unordered_set<std::string> fetch_ids(output_ids.begin(), output_ids.end());
  // apply common passes
  ProgramPass::Apply(program, fetch_ids, target, passes.first);

  // get original program size
  auto origin_size = program->size();
  // get original output
  auto origin_out = RunProgram(*program, target, input_ids, output_ids, seed, print_tensor);

  // apply fused passes
  ProgramPass::Apply(program, fetch_ids, target, passes.second);

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

}  // namespace cinn::frontend

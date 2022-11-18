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

#pragma once
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory.h>
#include <stdlib.h>

#include <functional>
#include <string>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/framework/tensor.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;
using ::cinn::hlir::framework::Shape;
using ::cinn::hlir::framework::Tensor;

void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Packaging matmul as a unified format required by the CheckResult interface
void target_func_matmul(const std::vector<float*>& inputs,
                        const std::vector<float*>& outputs,
                        const std::vector<std::vector<int>>& input_shapes,
                        const std::vector<std::vector<int>>& output_shapes) {
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(input_shapes.size(), 2);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(output_shapes.size(), 1);
  CHECK_EQ(input_shapes[0].size(), 2);
  CHECK_EQ(input_shapes[1].size(), 2);
  CHECK_EQ(output_shapes[0].size(), 2);
  CHECK_EQ(input_shapes[0][1], input_shapes[1][0]);
  CHECK_EQ(input_shapes[0][0], output_shapes[0][0]);
  CHECK_EQ(input_shapes[1][1], output_shapes[0][1]);
  int M = input_shapes[0][0];
  int N = input_shapes[1][1];
  int K = input_shapes[0][1];
  naive_matmul(inputs[0], inputs[1], outputs[0], M, N, K);
}

void MemoryCopy(const float* src, float* dst, int numel, std::string type) {
#ifdef CINN_WITH_CUDA
  if (type == "DeviceToHost") {
    cudaMemcpy(dst, src, numel * sizeof(float), cudaMemcpyDeviceToHost);
    return;
  } else if (type == "HostToDevice") {
    cudaMemcpy(dst, src, numel * sizeof(float), cudaMemcpyHostToDevice);
    return;
  }
#endif
  if (type == "HostToHost") {
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = src[i];
    }
  } else {
    CHECK(false) << "Unknown memory copy type";
  }
}

void AddDataToScope(Scope* scope,
                    const common::Target& target,
                    float* data_ptr,
                    int numel,
                    std::string name,
                    const std::vector<int>& shape) {
  auto* var    = scope->Var<Tensor>(name);
  auto& tensor = absl::get<Tensor>(*var);
  CHECK(shape.size());
  tensor->Resize(Shape(shape));
  auto* tgt_data_ptr = tensor->mutable_data<float>(target);
  std::string mem_cpy_type;
#ifdef CINN_WITH_CUDA
  mem_cpy_type = "HostToDevice";
#else
  mem_cpy_type = "HostToHost";
#endif
  MemoryCopy(data_ptr, tgt_data_ptr, numel, mem_cpy_type);
}

void InstantiateScope(Scope* scope,
                      const common::Target& target,
                      const std::vector<float*> input_data_ptrs,
                      const std::vector<float*> output_data_ptrs,
                      const std::vector<int> input_numels,
                      const std::vector<int> output_numels,
                      const std::vector<std::string>& input_names,
                      const std::vector<std::string>& output_names,
                      const std::vector<std::vector<int>>& input_shapes,
                      const std::vector<std::vector<int>>& output_shapes) {
  CHECK_EQ(input_names.size(), input_data_ptrs.size());
  CHECK_EQ(input_names.size(), input_numels.size());
  CHECK_EQ(input_names.size(), input_shapes.size());
  CHECK_EQ(output_names.size(), output_data_ptrs.size());
  CHECK_EQ(output_names.size(), output_numels.size());
  CHECK_EQ(output_names.size(), output_shapes.size());

  // input
  for (int i = 0; i < input_names.size(); ++i) {
    AddDataToScope(scope, target, input_data_ptrs[i], input_numels[i], input_names[i], input_shapes[i]);
  }
  // output
  for (int i = 0; i < output_names.size(); ++i) {
    AddDataToScope(scope, target, output_data_ptrs[i], output_numels[i], output_names[i], output_shapes[i]);
  }
}

/* @brief: Unified signature format as the target function for comparison.
 * @params-1: Pointers to the memory of input data, each input corresponds to a pointer.
 * @params-2: Pointers to the memory of output data, each output corresponds to a pointer.
 * @params-3: Shapes of the input data, each input corresponds to a std::vector<int>.
 * @params-4: Shapes of the output data, each output corresponds to a std::vector<int>.
 */
using target_func_type = void (*)(const std::vector<float*>&,
                                  const std::vector<float*>&,
                                  const std::vector<std::vector<int>>&,
                                  const std::vector<std::vector<int>>&);
/* @brief: Function pointer of executable code compiled by CINN.
 * @params-1: Pointers to all arguments, including input and output.
 * @params-2: The number of Arguments.
 */
using test_func_type = void (*)(void**, int32_t);

/* @brief: Interface for checking function correctness.
 * @params-1: Function pointer of the function to be tested.
 * @params-2: Target function pointer for comparison.
 * @params-3: Names of input data.
 * @params-4: Names of output data.
 * @params-5: Shapes of the input data, each input corresponds to a std::vector<int>.
 * @params-6: Shapes of the output data, each output corresponds to a std::vector<int>.
 * @params-7: The Target expressing computing platform and architecture of the function to be tested.
 */
void CheckResult(test_func_type test_func,
                 target_func_type target_func,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names,
                 const std::vector<std::vector<int>>& input_shapes,
                 const std::vector<std::vector<int>>& output_shapes,
                 const common::Target& target) {
  CHECK(input_names.size());
  CHECK(output_names.size());
  CHECK_EQ(input_names.size(), input_shapes.size());
  CHECK_EQ(output_names.size(), output_shapes.size());

  // Initialize data
  std::vector<float*> input_data_ptrs(input_names.size());
  std::vector<int> input_data_numels(input_shapes.size());
  for (int i = 0; i < input_shapes.size(); ++i) {
    int data_numel = 1;
    for (int n : input_shapes[i]) {
      data_numel *= n;
    }
    input_data_numels[i] = data_numel;
    input_data_ptrs[i]   = reinterpret_cast<float*>(malloc(data_numel * sizeof(float)));
    for (int j = 0; j < data_numel; ++j) {
      input_data_ptrs[i][j] = (rand() * 1.f) / RAND_MAX;
    }
  }
  std::vector<float*> output_data_ptrs(output_names.size());
  std::vector<float*> target_output_data_ptrs(output_names.size());
  std::vector<int> output_data_numels(output_shapes.size());
  for (int i = 0; i < output_shapes.size(); ++i) {
    int data_numel = 1;
    for (int n : output_shapes[i]) {
      data_numel *= n;
    }
    output_data_numels[i] = data_numel;
    output_data_ptrs[i]   = reinterpret_cast<float*>(malloc(data_numel * sizeof(float)));
    memset(output_data_ptrs[i], 0, data_numel * sizeof(float));
    target_output_data_ptrs[i] = reinterpret_cast<float*>(malloc(data_numel * sizeof(float)));
    memset(target_output_data_ptrs[i], 0, data_numel * sizeof(float));
  }

  Scope scope;
  InstantiateScope(&scope,
                   target,
                   input_data_ptrs,
                   output_data_ptrs,
                   input_data_numels,
                   output_data_numels,
                   input_names,
                   output_names,
                   input_shapes,
                   output_shapes);

  // Create Instruction and run
  Instruction instr(target, &scope, input_names, output_names);
  CHECK(test_func);
  instr.SetLoweredFunc(reinterpret_cast<void*>(test_func));
  // should call Finalize explicitly before Run
  ASSERT_DEATH(instr.Run(), "");
  instr.Finalize();
  instr.Run();

  // Get data
  for (int i = 0; i < output_names.size(); ++i) {
    const float* result_ptr = scope.GetTensor(output_names[i])->data<float>();
    std::string mem_cpy_type;
#ifdef CINN_WITH_CUDA
    mem_cpy_type = "DeviceToHost";
#else
    mem_cpy_type = "HostToHost";
#endif
    MemoryCopy(result_ptr, output_data_ptrs[i], output_data_numels[i], mem_cpy_type);
  }

  // Calculate target result
  target_func(input_data_ptrs, target_output_data_ptrs, input_shapes, output_shapes);

  // Check result
  for (int i = 0; i < output_shapes.size(); ++i) {
    for (int j = 0; j < output_data_numels[i]; ++j) {
      ASSERT_NEAR(output_data_ptrs[i][j], target_output_data_ptrs[i][j], 1e-5);
    }
  }

  // Free memory
  for (auto ptr : input_data_ptrs) {
    free(ptr);
  }
  for (auto ptr : output_data_ptrs) {
    free(ptr);
  }
  for (auto ptr : target_output_data_ptrs) {
    free(ptr);
  }
}

}  // namespace auto_schedule
}  // namespace cinn

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

#pragma once

#include <gtest/gtest.h>

#include <random>

#include "cinn/frontend/decomposer/use_decomposer.h"
#include "cinn/frontend/decomposer_registry.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn::frontend {

using CPUKernelFunc = std::function<void(const std::vector<size_t>& lengths, const std::vector<void*>& ptrs)>;

template <typename T, typename Alloc = std::allocator<T>>
std::ostream& operator<<(std::ostream& os, const std::vector<T, Alloc>& vec) {
  os << "{";
  bool is_first = true;
  for (auto e : vec) {
    if (is_first) {
      is_first = false;
    } else {
      os << ", ";
    }
    os << e;
  }
  os << "}\n";
  return os;
}

Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

template <typename T>
void InitRandomVector(std::vector<T>* vec, size_t numel, T low = 0, T high = 1, T precision = 1e-5) {
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<double> dist(low, high);

  vec->resize(numel);
  for (size_t i = 0; i < numel; ++i) {
    T value    = static_cast<T>(dist(engine));
    int coeff  = static_cast<int>(value / precision);
    vec->at(i) = precision * static_cast<T>(coeff);
  }
}

template <typename T>
void CopyFromVector(const std::vector<T>& vec, hlir::framework::Tensor tensor, Target target) {
  auto* data = tensor->mutable_data<T>(target);

  size_t numel = tensor->shape().numel();
  EXPECT_EQ(vec.size(), numel);

#ifdef CINN_WITH_CUDA
  cudaMemcpy(data, vec.data(), numel * sizeof(T), cudaMemcpyHostToDevice);
#else
  std::copy(vec.begin(), vec.end(), data);
#endif
}

template <typename T>
void CopyToVector(const hlir::framework::Tensor tensor, std::vector<T>* vec) {
  auto* data = tensor->data<T>();

  size_t numel = tensor->shape().numel();
  vec->resize(numel);

#ifdef CINN_WITH_CUDA
  cudaMemcpy(vec->data(), data, numel * sizeof(T), cudaMemcpyDeviceToHost);
#else
  for (size_t i = 0; i < numel; ++i) {
    vec->at(i) = data[i];
  }
#endif
}

template <typename T>
void CheckOutput(const std::vector<T>& actual, const std::vector<T>& expect, float atol = 1e-8, float rtol = 1e-5) {
  CHECK_EQ(actual.size(), expect.size());

  auto allclose = [](T a, T e, float atol, float rtol) { return abs(a - e) <= (atol + rtol * abs(e)); };

  float max_diff = 0.0f;
  int offset     = 0;
  int num_diffs  = 0;

  size_t numel = actual.size();
  for (size_t i = 0; i < numel; ++i) {
    if (!allclose(actual[i], expect[i], atol, rtol)) {
      float absolute_diff = abs((actual[i] - expect[i]));
      float relative_diff = abs(absolute_diff / expect[i]);
      if (relative_diff > max_diff) {
        max_diff = relative_diff;
        offset   = i;
      }
      num_diffs += 1;
      VLOG(4) << "- i=" << i << ", " << std::setprecision(8) << actual[i] << " (actual) vs " << std::setprecision(8)
              << expect[i] << " (expect), relative_diff=" << relative_diff << ", absolute_diff=" << absolute_diff;
    }
  }
  VLOG(1) << "- Total " << num_diffs << " different results, offset=" << offset << ", " << actual[offset]
          << " (actual) vs " << expect[offset] << " (expect), maximum_relative_diff=" << max_diff
          << " (absolute_diff=" << abs((actual[offset] - expect[offset])) << ")";
  ASSERT_EQ(num_diffs, 0);
}

template <typename T>
void ComputeReferenceCpu(const std::vector<std::vector<T>>& input_vecs,
                         const std::vector<std::vector<T>>& output_vecs,
                         std::vector<std::vector<T>>* output_refs,
                         CPUKernelFunc cpu_kernel_func) {
  output_refs->resize(output_vecs.size());
  for (size_t i = 0; i < output_vecs.size(); ++i) {
    output_refs->at(i).resize(output_vecs[i].size());
  }

  // Prepare the arguments for reference.
  // For different operations, the needed parameters maybe different.
  size_t n = input_vecs[0].size();
  std::vector<size_t> lengths;
  lengths.push_back(n);

  std::vector<void*> ptrs(input_vecs.size() + output_refs->size());
  for (size_t i = 0; i < input_vecs.size(); ++i) {
    ptrs[i] = const_cast<void*>(static_cast<const void*>(input_vecs[i].data()));
  }
  for (size_t i = 0; i < output_refs->size(); ++i) {
    ptrs[input_vecs.size() + i] = output_refs->at(i).data();
  }
  cpu_kernel_func(lengths, ptrs);
}

void RunDecomposer(Program* prog, const Target& target) {
  VLOG(1) << "===================== Before Decomposition =====================";
  for (int i = 0; i < prog->size(); i++) {
    VLOG(1) << "instruction: " << (*prog)[i];
  }
  ProgramPass::Apply(prog, target, {"Decomposer"});
  VLOG(1) << "===================== After Decomposition =====================";
  for (int i = 0; i < prog->size(); i++) {
    VLOG(1) << "instruction: " << (*prog)[i];
  }
}

template <typename T>
void RunAndCheckShape(NetBuilder& builder,
                      const std::vector<std::string>& input_names,
                      const std::vector<std::string>& output_names,
                      const std::vector<std::vector<int>>& output_shapes,
                      std::vector<std::vector<T>>* input_vecs  = nullptr,
                      std::vector<std::vector<T>>* output_vecs = nullptr,
                      T low                                    = 0,
                      T high                                   = 1) {
  auto prog     = builder.Build();
  Target target = GetTarget();
  RunDecomposer(&prog, target);
  auto graph = std::make_shared<hlir::framework::Graph>(prog, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  auto runtime_program = gc.Build();
  std::vector<std::vector<T>> input_vecs_internal;
  std::vector<std::vector<T>>* input_vecs_ptr = input_vecs ? input_vecs : &input_vecs_internal;
  for (size_t i = 0; i < input_names.size(); ++i) {
    scope->Var<hlir::framework::Tensor>(input_names[i]);
    auto tensor = scope->GetTensor(input_names[i]);

    std::vector<T> vec;
    InitRandomVector<T>(&vec, tensor->shape().numel(), low, high);
    CopyFromVector<T>(vec, tensor, target);
    input_vecs_ptr->push_back(vec);
  }
  runtime_program->Execute();

  for (size_t i = 0; i < output_names.size(); ++i) {
    auto tensor = scope->GetTensor(output_names[i]);
    CHECK_EQ(tensor->shape().data() == output_shapes[i], true)
        << "The " << i << "-th shape is expected to be " << output_shapes[i];
    if (output_vecs) {
      std::vector<T> vec;
      CopyToVector<T>(tensor, &vec);
      output_vecs->push_back(vec);
    }
  }
}

template <typename T>
void RunAndCheck(NetBuilder& builder,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names,
                 const std::vector<std::vector<int>>& output_shapes,
                 CPUKernelFunc cpu_kernel_func,
                 T low      = 0,
                 T high     = 1,
                 float atol = 1e-8,
                 float rtol = 1e-5) {
  std::vector<std::vector<T>> input_vecs;
  std::vector<std::vector<T>> output_vecs;
  RunAndCheckShape<T>(builder, input_names, output_names, output_shapes, &input_vecs, &output_vecs, low, high);

  std::vector<std::vector<T>> output_refs;
  ComputeReferenceCpu<T>(input_vecs, output_vecs, &output_refs, cpu_kernel_func);

  for (size_t i = 0; i < output_vecs.size(); ++i) {
    VLOG(1) << "Check the " << i << "-th output, name=" << output_names[i] << ", shape=" << output_shapes[i];
    CheckOutput<T>(output_vecs[i], output_refs[i], atol, rtol);
  }
}

}  // namespace cinn::frontend

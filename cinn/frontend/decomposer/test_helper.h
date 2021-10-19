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

Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

template <typename T>
void InitRandomVector(std::vector<T> *vec, size_t numel) {
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  vec->resize(numel);
  for (size_t i = 0; i < numel; ++i) {
    vec->at(i) = static_cast<T>(dist(engine));
  }
}

template <typename T>
void CopyFromVector(const std::vector<T> &vec, hlir::framework::Tensor tensor, Target target) {
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
void RunProgram(NetBuilder& builder, const std::vector<std::string>& input_names) {
  auto prog     = builder.Build();
  Target target = GetTarget();
  LOG(INFO) << "===================== Before Decomposition =====================";
  for (int i = 0; i < prog.size(); i++) {
    LOG(INFO) << "instruction: " << prog[i];
  }
  ProgramPass::Apply(&prog, target, {"Decomposer"});
  LOG(INFO) << "===================== After Decomposition =====================";
  for (int i = 0; i < prog.size(); i++) {
    LOG(INFO) << "instruction: " << prog[i];
  }
  auto graph = std::make_shared<hlir::framework::Graph>(prog, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  auto runtime_program = gc.Build();
  for (auto& name : input_names) {
    scope->Var<hlir::framework::Tensor>(name);
    auto tensor = scope->GetTensor(name);
  
    std::vector<T> vec;
    InitRandomVector<T>(&vec, tensor->shape().numel());
    CopyFromVector<T>(vec, tensor, target);
  }
  runtime_program->Execute();
}

}  // namespace cinn::frontend

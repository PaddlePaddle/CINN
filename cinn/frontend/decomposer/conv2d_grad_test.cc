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
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif
#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/decomposer/test_helper.h"
#include "cinn/frontend/decomposer/use_decomposer.h"
#include "cinn/frontend/decomposer_registry.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace frontend {
namespace {

TEST(nn, CONV_GRAD) {
  int n = 32, ic = 16, h = 32, w = 32;
  int fh = 3, fw = 3;
  int oc = 32;

  std::vector<int> strides   = {1, 1};
  std::vector<int> paddings  = {1, 1};
  std::vector<int> dilations = {1, 1};

  NetBuilder net_builder("net_builder");
  {
    // create input
    auto dy     = net_builder.CreateInput(Float(32), {n, oc, h, w}, "dy");
    auto x      = net_builder.CreateInput(Float(32), {n, ic, h, w}, "x");
    auto weight = net_builder.CreateInput(Float(32), {oc, ic, fh, fw}, "weight");
    // add batch norm train
    auto outputs = net_builder.conv2d_grad(dy, x, weight, strides, paddings, dilations);
  }
  // build program
  auto program = net_builder.Build();
  auto target  = common::DefaultNVGPUTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto run_program = gc.Build();

  // set input
  std::vector<float> x(n * ic * h * w), weight(oc * ic * fh * fw), dy(n * oc * h * w);
  InitRandomVector(&x, n * ic * h * w);
  InitRandomVector(&weight, oc * ic * fh * fw);
  InitRandomVector(&dy, n * oc * h * w);

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {{"x", x}, {"weight", weight}, {"dy", dy}};

  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    auto* data  = tensor->mutable_data<float>(target);
    LOG(INFO) << input.first << " " << tensor->shape().numel();
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();
}

}  // namespace
}  // namespace frontend
}  // namespace cinn

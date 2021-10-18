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
#include "cinn/frontend/decomposer/use_decomposer.h"
#include "cinn/frontend/decomposer_registry.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace frontend {
namespace {

TEST(nn, BATCH_NORM_TRAIN) {
  // parameter
  int n = 8, c = 16, h = 14, w = 14;
  NetBuilder net_builder("net_builder");
  {
    // create input
    auto x            = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto scale        = net_builder.CreateInput(Float(32), {c}, "scale");
    auto bias         = net_builder.CreateInput(Float(32), {c}, "bias");
    auto running_mean = net_builder.CreateInput(Float(32), {c}, "running_mean");
    auto running_var  = net_builder.CreateInput(Float(32), {c}, "running_var");

    // add batch norm train
    auto outputs = net_builder.batch_norm_train(x, scale, bias, running_mean, running_var);
  }
  // build program
  auto program = net_builder.Build();

  auto target = ::cinn::common::DefaultHostTarget();
  // ProgramPass::Apply(&program, target, {"Decomposer"});
  CinnBuilder cinn_builder("cinn_builder");
  {
    auto x            = cinn_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto scale        = cinn_builder.CreateInput(Float(32), {c}, "scale");
    auto bias         = cinn_builder.CreateInput(Float(32), {c}, "bias");
    auto running_mean = cinn_builder.CreateInput(Float(32), {c}, "running_mean");
    auto running_var  = cinn_builder.CreateInput(Float(32), {c}, "running_var");
  }
  // CinnBuilder cinn_builder;
  absl::flat_hash_map<std::string, Variable> variable_map;
  DecomposerContext context(&cinn_builder, &variable_map);
  auto decomposer = InstrDecomposerRegistry::Global()->Get("batch_norm_train", target);

  decomposer->Run(program[0], context);
  auto new_program = cinn_builder.Build();

  auto graph = std::make_shared<hlir::framework::Graph>(new_program, target);
  auto nodes = graph->nodes();

  /*
  for(auto node : nodes) {
    for(auto link : node->outlinks()) {
      std::cout<<" " << link.get()->sink()->id();
    }
    std::cout << " -> ";
    std::cout << node->id() << " -> ";
    for(auto link : node->inlinks()) {
      std::cout<<" " << link.get()->source()->id();
    }
    std::cout << std::endl;
  }
  */
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  auto run_program = gc.Build();
}

TEST(nn, BATCH_NORM_GRAD) {
  // parameter
  int n = 8, c = 16, h = 14, w = 14;
  NetBuilder net_builder("net_builder");
  {
    // create input
    auto x         = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto dy        = net_builder.CreateInput(Float(32), {n, c, h, w}, "dy");
    auto scale     = net_builder.CreateInput(Float(32), {c}, "scale");
    auto save_mean = net_builder.CreateInput(Float(32), {c}, "save_mean");
    auto save_var  = net_builder.CreateInput(Float(32), {c}, "save_var");

    // add batch norm train
    auto outputs = net_builder.batch_norm_grad(x, dy, scale, save_mean, save_var);
  }
  // build program
  auto program = net_builder.Build();

  auto target = ::cinn::common::DefaultHostTarget();
  CinnBuilder cinn_builder("cinn_builder");
  {
    auto x         = cinn_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto dy        = cinn_builder.CreateInput(Float(32), {n, c, h, w}, "dy");
    auto scale     = cinn_builder.CreateInput(Float(32), {c}, "scale");
    auto save_mean = cinn_builder.CreateInput(Float(32), {c}, "save_mean");
    auto save_var  = cinn_builder.CreateInput(Float(32), {c}, "save_var");
  }
  absl::flat_hash_map<std::string, Variable> variable_map;
  DecomposerContext context(&cinn_builder, &variable_map);
  auto decomposer = InstrDecomposerRegistry::Global()->Get("batch_norm_grad", target);

  decomposer->Run(program[0], context);
  auto new_program = cinn_builder.Build();

  auto graph = std::make_shared<hlir::framework::Graph>(new_program, target);
  auto nodes = graph->nodes();

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto run_program = gc.Build();

  /*
  for(auto node : nodes) {
    for(auto link : node->outlinks()) {
      std::cout<<" " << link.get()->sink()->id();
    }
    std::cout << " -> ";
    std::cout << node->id() << " -> ";
    for(auto link : node->inlinks()) {
      std::cout<<" " << link.get()->source()->id();
    }
    std::cout << std::endl;
  }
  */
}

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
    auto x = net_builder.CreateInput(Float(32), {n, ic, h, w}, "x");
    auto weight = net_builder.CreateInput(Float(32), {oc, ic, fh, fw}, "weight");
    auto dy = net_builder.CreateInput(Float(32), {n, oc, h, w}, "y");
    // add batch norm train
    auto outputs = net_builder.conv2d_grad(x, weight, dy, strides, paddings, dilations);
  }
  // build program
  auto program = net_builder.Build();

  auto target = ::cinn::common::DefaultNVGPUTarget();
  CinnBuilder cinn_builder("cinn_builder");
  {
    // create input
    auto x = cinn_builder.CreateInput(Float(32), {n, ic, h, w}, "x");
    auto weight = cinn_builder.CreateInput(Float(32), {oc, ic, fh, fw}, "weight");
    auto dy = cinn_builder.CreateInput(Float(32), {n, oc, h, w}, "dy");
  }

  absl::flat_hash_map<std::string, Variable> variable_map;
  DecomposerContext context(&cinn_builder, &variable_map);
  auto decomposer = InstrDecomposerRegistry::Global()->Get("conv2d_grad", target);

  decomposer->Run(program[0], context);
  auto new_program = cinn_builder.Build();

  auto graph = std::make_shared<hlir::framework::Graph>(new_program, target);
  auto nodes = graph->nodes();

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto run_program = gc.Build();
}

}  // namespace
}  // namespace frontend
}  // namespace cinn

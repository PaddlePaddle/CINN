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

template <typename FuncType>
void loop(FuncType func, const int n, const int c, const int h, const int w) {
  for (int idx = 0; idx < n; ++idx) {
    for (int idy = 0; idy < c; ++idy) {
      for (int idz = 0; idz < h; ++idz) {
        for (int ida = 0; ida < w; ++ida) {
          func(idx, idy, idz, ida);
        }
      }
    }
  }
}

template <typename T>
void cpu_run_batch_norm_train(const std::vector<T>& x,
                              const std::vector<T>& scale,
                              const std::vector<T>& bias,
                              const std::vector<T>& running_mean,
                              const std::vector<T>& running_var,
                              const int n,
                              const int c,
                              const int h,
                              const int w,
                              std::vector<T>* y,
                              std::vector<T>* sum,
                              std::vector<T>* mean,
                              std::vector<T>* diff,
                              std::vector<T>* diff2,
                              std::vector<T>* sum_diff2,
                              std::vector<T>* mean_diff2,
                              std::vector<T>* var,
                              std::vector<T>* std,
                              std::vector<T>* mul_scale,
                              std::vector<T>* new_running_mean,
                              std::vector<T>* new_running_var,
                              const float epsilon  = 1e-6,
                              const float momentum = 0.9f) {
  // sum
  memset(sum->data(), 0, sizeof(T) * c);
  auto func_sum = [=](int idx, int idy, int idz, int ida) {
    sum->at(idy) += x[idx * c * h * w + idy * h * w + idz * w + ida];
  };
  loop(func_sum, n, c, h, w);
  // mean
  for (int idx = 0; idx < c; ++idx) {
    mean->at(idx) = sum->at(idx) / float(n * h * w);
  }

  // diff
  auto func_diff = [=](int idx, int idy, int idz, int ida) {
    auto val = x[idx * c * h * w + idy * h * w + idz * w + ida] - mean->at(idy);
    diff->at(idx * c * h * w + idy * h * w + idz * w + ida)  = val;
    diff2->at(idx * c * h * w + idy * h * w + idz * w + ida) = val * val;
  };
  loop(func_diff, n, c, h, w);

  // sum diff2
  memset(sum_diff2->data(), 0, sizeof(T) * c);
  auto func_sum_diff2 = [=](int idx, int idy, int idz, int ida) {
    sum_diff2->at(idy) += diff2->at(idx * c * h * w + idy * h * w + idz * w + ida);
  };
  loop(func_sum_diff2, n, c, h, w);

  // var
  memset(var->data(), 0, sizeof(T) * c);
  for (int idx = 0; idx < c; ++idx) {
    mean_diff2->at(idx) = sum_diff2->at(idx) / float(n * h * w);
    var->at(idx)        = sqrt(mean_diff2->at(idx)) + epsilon;
  }

  // compute output
  auto func_y = [=](int idx, int idy, int idz, int ida) {
    std->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        diff->at(idx * c * h * w + idy * h * w + idz * w + ida) / var->at(idy);
    mul_scale->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        std->at(idx * c * h * w + idy * h * w + idz * w + ida) * scale[idy];
    y->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        mul_scale->at(idx * c * h * w + idy * h * w + idz * w + ida) + bias[idy];
  };
  loop(func_y, n, c, h, w);

  // update runnning
  for (int idx = 0; idx < c; ++idx) {
    new_running_mean->at(idx) = running_mean[idx] * momentum + mean->at(idx) * (1 - momentum);
    new_running_var->at(idx)  = running_var[idx] * momentum + var->at(idx) * (1 - momentum);
  }
}

TEST(nn, BATCH_NORM_TRAIN) {
  // parameter
  int n = 4, c = 16, h = 4, w = 4;
  NetBuilder net_builder("net_builder_batch_norm_train");
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
  // auto target = ::cinn::common::DefaultNVGPUTarget();
  CinnBuilder cinn_builder("cinn_builder_batch_norm_train");
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
  // hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto nodes = std::get<0>(graph->topological_order());

  /*
  for (auto& node : nodes) {
    for (auto link : node->inlinks()) {
      std::cerr << link->source()->id() << " ";
    }
    std::cerr << " -> " << node->id() << " -> ";
    for (auto link : node->outlinks()) {
      std::cerr << link->sink()->id() << " ";
    }
    std::cerr << std::endl;
  }
  */

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  auto run_program = gc.Build();

  // set input
  std::vector<float> x(n * c * h * w), scale(c), bias(c), running_mean(c), running_var(c);
  std::vector<float> y(n * c * h * w), sum(c), mean(c), diff(n * c * h * w), diff2(n * c * h * w), sum_diff2(c),
      mean_diff2(c), var(c), std(n * c * h * w), mul_scale(n * c * h * w), new_running_mean(c), new_running_var(c);

  InitRandomVector(&x, n * c * h * w);
  InitRandomVector(&scale, c);
  InitRandomVector(&bias, c);
  InitRandomVector(&running_mean, c);
  InitRandomVector(&running_var, c);

  cpu_run_batch_norm_train(x,
                           scale,
                           bias,
                           running_mean,
                           running_var,
                           n,
                           c,
                           h,
                           w,
                           &y,
                           &sum,
                           &mean,
                           &diff,
                           &diff2,
                           &sum_diff2,
                           &mean_diff2,
                           &var,
                           &std,
                           &mul_scale,
                           &new_running_mean,
                           &new_running_var);

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {
      {"x", x}, {"scale", scale}, {"bias", bias}, {"running_mean", running_mean}, {"running_var", running_var}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    auto* data  = tensor->mutable_data<float>(target);
    memcpy(data, input.second.data(), tensor->shape().numel() * sizeof(float));
    // LOG(INFO) << input.first << " " << tensor->shape().numel();
  }
  std::vector<std::pair<std::string, std::vector<float>>> outputs = {{"var_18", sum},
                                                                     {"var_19", mean},
                                                                     {"var_21", diff},
                                                                     {"var_23", diff2},
                                                                     {"var_24", sum_diff2},
                                                                     {"var_25", mean_diff2},
                                                                     {"var_27", var},
                                                                     {"var_31", std},
                                                                     {"var_40", new_running_mean},
                                                                     {"var_43", new_running_var},
                                                                     {"var_33", y}};

  run_program->Execute();

  for (auto& output : outputs) {
    auto tensor = scope->GetTensor(output.first);
    auto* data  = tensor->data<float>();

    LOG(INFO) << output.first << " " << tensor->shape().numel();
    for (int idx = 0; idx < tensor->shape().numel(); ++idx) {
      ASSERT_LT(abs((data[idx] - output.second[idx]) / data[idx]), 1e-4);
    }
  }
}

}  // namespace
}  // namespace frontend
}  // namespace cinn

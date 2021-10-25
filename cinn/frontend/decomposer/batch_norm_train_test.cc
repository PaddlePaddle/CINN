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
                              const std::vector<T>& moving_mean,
                              const std::vector<T>& moving_variance,
                              const int n,
                              const int c,
                              const int h,
                              const int w,
                              std::vector<T>* sum,
                              std::vector<T>* mean,
                              std::vector<T>* sum_square,
                              std::vector<T>* mean_square,
                              std::vector<T>* variance,
                              std::vector<T>* std_variance,
                              std::vector<T>* y,
                              std::vector<T>* new_moving_mean,
                              std::vector<T>* new_moving_variance,
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

  // square
  memset(sum_square->data(), 0, sizeof(T) * c);
  auto func_sum_square = [=](int idx, int idy, int idz, int ida) {
    sum_square->at(idy) +=
        x[idx * c * h * w + idy * h * w + idz * w + ida] * x[idx * c * h * w + idy * h * w + idz * w + ida];
  };
  loop(func_sum_square, n, c, h, w);
  //
  for (int idx = 0; idx < c; ++idx) {
    mean_square->at(idx) = sum_square->at(idx) / float(n * h * w);
  }

  // sum diff2
  for (int idx = 0; idx < c; ++idx) {
    variance->at(idx)     = mean_square->at(idx) - (mean->at(idx) * mean->at(idx));
    std_variance->at(idx) = sqrt(variance->at(idx) + epsilon);
  }

  // compute output
  auto func_y = [=](int idx, int idy, int idz, int ida) {
    y->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        (x[idx * c * h * w + idy * h * w + idz * w + ida] - mean->at(idy)) / std_variance->at(idy) * scale[idy] +
        bias[idy];
  };
  loop(func_y, n, c, h, w);

  // update runnning
  for (int idx = 0; idx < c; ++idx) {
    new_moving_mean->at(idx)     = moving_mean[idx] * momentum + mean->at(idx) * (1 - momentum);
    new_moving_variance->at(idx) = moving_variance[idx] * momentum + variance->at(idx) * (1 - momentum);
  }
}

TEST(nn, BATCH_NORM_TRAIN) {
  // parameter
  int n = 4, c = 16, h = 4, w = 4;
  NetBuilder net_builder("net_builder_batch_norm_train");
  {
    // create input
    auto x               = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto scale           = net_builder.CreateInput(Float(32), {c}, "scale");
    auto bias            = net_builder.CreateInput(Float(32), {c}, "bias");
    auto moving_mean     = net_builder.CreateInput(Float(32), {c}, "moving_mean");
    auto moving_variance = net_builder.CreateInput(Float(32), {c}, "moving_variance");

    // add batch norm train
    auto outputs = net_builder.batch_norm_train(x, scale, bias, moving_mean, moving_variance);
  }
  // build program
  auto program = net_builder.Build();

  auto target = GetTarget();
  CinnBuilder cinn_builder("cinn_builder_batch_norm_train");
  {
    auto x               = cinn_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto scale           = cinn_builder.CreateInput(Float(32), {c}, "scale");
    auto bias            = cinn_builder.CreateInput(Float(32), {c}, "bias");
    auto moving_mean     = cinn_builder.CreateInput(Float(32), {c}, "moving_mean");
    auto moving_variance = cinn_builder.CreateInput(Float(32), {c}, "moving_variance");
  }
  // CinnBuilder cinn_builder;
  absl::flat_hash_map<std::string, Variable> variable_map;
  DecomposerContext context(&cinn_builder, &variable_map);
  auto decomposer = InstrDecomposerRegistry::Global()->Get("batch_norm_train", target);

  decomposer->Run(program[0], context);
  auto new_program = cinn_builder.Build();

  auto graph = std::make_shared<hlir::framework::Graph>(new_program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto nodes = std::get<0>(graph->topological_order());
  LOG(INFO) << graph->Visualize();

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  auto run_program = gc.Build();

  // set input
  std::vector<float> x(n * c * h * w), scale(c), bias(c), moving_mean(c), moving_variance(c);
  std::vector<float> sum(c), mean(c), sum_square(c), mean_square(c), variance(c), std_variance(c);
  std::vector<float> y(n * c * h * w), new_moving_mean(c), new_moving_variance(c);

  InitRandomVector(&x, n * c * h * w);
  InitRandomVector(&scale, c);
  InitRandomVector(&bias, c);
  InitRandomVector(&moving_mean, c);
  InitRandomVector(&moving_variance, c);

  cpu_run_batch_norm_train(x,
                           scale,
                           bias,
                           moving_mean,
                           moving_variance,
                           n,
                           c,
                           h,
                           w,
                           &sum,
                           &mean,
                           &sum_square,
                           &mean_square,
                           &variance,
                           &std_variance,
                           &y,
                           &new_moving_mean,
                           &new_moving_variance);

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {
      {"x", x}, {"scale", scale}, {"bias", bias}, {"moving_mean", moving_mean}, {"moving_variance", moving_variance}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    auto* data  = tensor->mutable_data<float>(target);
    CopyFromVector(input.second, tensor, target);
  }

  std::vector<std::pair<std::string, std::vector<float>>> outputs = {{"new_moving_mean", new_moving_mean},
                                                                     {"new_moving_variance", new_moving_variance},
                                                                     {"batch_norm_train_output", y}};

  run_program->Execute();

  for (auto& output : outputs) {
    auto tensor = scope->GetTensor(output.first);
    std::vector<float> data(tensor->shape().numel());
    CopyToVector(tensor, &data);

    LOG(INFO) << output.first << " " << tensor->shape().numel();
    for (int idx = 0; idx < tensor->shape().numel(); ++idx) {
      ASSERT_LT(abs((data[idx] - output.second[idx]) / data[idx]), 1e-4);
    }
  }
}

}  // namespace
}  // namespace frontend
}  // namespace cinn

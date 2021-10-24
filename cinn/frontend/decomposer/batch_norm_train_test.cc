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
void cpu_run_batch_norm_train(const T* x,
                              const T* scale,
                              const T* bias,
                              const T* running_mean,
                              const T* running_var,
                              const int n,
                              const int c,
                              const int h,
                              const int w,
                              T* y,
                              T* sum,
                              T* mean,
                              T* diff,
                              T* diff2,
                              T* sum_diff2,
                              T* mean_diff2,
                              T* var,
                              T* std,
                              T* mul_scale,
                              T* new_running_mean,
                              T* new_running_var,
                              const float epsilon        = 1e-6,
                              const float running_factor = 0.99f) {
  // sum
  memset(sum, 0, sizeof(T) * c);
  auto func_sum = [=](int idx, int idy, int idz, int ida) {
    sum[idy] += x[idx * c * h * w + idy * h * w + idz * w + ida];
  };
  loop(func_sum, n, c, h, w);

  // mean
  for (int idx = 0; idx < c; ++idx) {
    mean[idx] = sum[idx] / float(n * h * w);
  }

  // diff
  auto func_diff = [=](int idx, int idy, int idz, int ida) {
    auto val                                             = x[idx * c * h * w + idy * h * w + idz * w + ida] - mean[idy];
    diff[idx * c * h * w + idy * h * w + idz * w + ida]  = val;
    diff2[idx * c * h * w + idy * h * w + idz * w + ida] = val * val;
  };
  loop(func_diff, n, c, h, w);

  // sum diff2
  memset(sum_diff2, 0, sizeof(T) * c);
  auto func_sum_diff2 = [=](int idx, int idy, int idz, int ida) {
    sum_diff2[idy] += diff2[idx * c * h * w + idy * h * w + idz * w + ida];
  };
  loop(func_sum_diff2, n, c, h, w);

  // var
  memset(var, 0, sizeof(T) * c);
  for (int idx = 0; idx < c; ++idx) {
    mean_diff2[idx] = sum_diff2[idx] / float(n * h * w);
    var[idx]        = sqrt(mean_diff2[idx]) + epsilon;
  }

  // compute output
  auto func_y = [=](int idx, int idy, int idz, int ida) {
    std[idx * c * h * w + idy * h * w + idz * w + ida] =
        diff[idx * c * h * w + idy * h * w + idz * w + ida] / (var[idy]);
    mul_scale[idx * c * h * w + idy * h * w + idz * w + ida] =
        std[idx * c * h * w + idy * h * w + idz * w + ida] * scale[idy];
    y[idx * c * h * w + idy * h * w + idz * w + ida] =
        mul_scale[idx * c * h * w + idy * h * w + idz * w + ida] + bias[idy];
  };
  loop(func_y, n, c, h, w);

  // update runnning
  for (int idx = 0; idx < c; ++idx) {
    new_running_mean[idx] = running_mean[idx] * running_factor + mean[idx] * (1 - running_factor);
    new_running_var[idx]  = running_var[idx] * running_factor + var[idx] * (1 - running_factor);
  }
}

template <typename T>
void random(T* value, int num) {
  for (int idx = 0; idx < num; ++idx) {
    *value++ = rand() / 100000.0f;
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

  // auto target = ::cinn::common::DefaultHostTarget();
  auto target = ::cinn::common::DefaultNVGPUTarget();
  // ProgramPass::Apply(&program, target, {"Decomposer"});
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
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto nodes = std::get<0>(graph->topological_order());

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

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  auto run_program = gc.Build();

  // set input
  float *x = new float[n * c * h * w], *scale = new float[c], *bias = new float[c], *running_mean = new float[c],
        *running_var = new float[c];
  float *y = new float[n * c * h * w], *sum = new float[c], *mean = new float[c], *diff = new float[n * c * h * w],
        *diff2 = new float[n * c * h * w], *sum_diff2 = new float[c], *mean_diff2 = new float[c], *var = new float[c],
        *std = new float[n * c * h * w], *mul_scale = new float[n * c * h * w], *new_running_mean = new float[c],
        *new_running_var = new float[c];

  random(x, n * c * h * w);
  random(scale, c);
  random(bias, c);
  random(running_mean, c);
  random(running_var, c);

  cpu_run_batch_norm_train(x,
                           scale,
                           bias,
                           running_mean,
                           running_var,
                           n,
                           c,
                           h,
                           w,
                           y,
                           sum,
                           mean,
                           diff,
                           diff2,
                           sum_diff2,
                           mean_diff2,
                           var,
                           std,
                           mul_scale,
                           new_running_mean,
                           new_running_var);

  std::vector<std::pair<std::string, float*>> inputs = {
      {"x", x}, {"scale", scale}, {"bias", bias}, {"running_mean", running_mean}, {"running_var", running_var}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    auto* data  = tensor->mutable_data<float>(target);
    memcpy(data, input.second, tensor->shape().numel() * sizeof(float));
    // LOG(INFO) << input.first << " " << tensor->shape().numel();
  }
  std::vector<std::pair<std::string, float*>> outputs = {
      {"var_40", new_running_mean}, {"var_43", new_running_var}, {"var_33", y}};

  run_program->Execute();

  for (auto& output : outputs) {
    auto tensor = scope->GetTensor(output.first);
    auto* data  = tensor->data<float>();

    LOG(INFO) << output.first << " " << tensor->shape().numel();
    for (int idx = 0; idx < tensor->shape().numel(); ++idx) {
      ASSERT_LT(abs((data[idx] - output.second[idx]) / data[idx]), 2e-5);
    }
  }

  delete x;
  delete scale;
  delete bias;
  delete running_mean;
  delete running_var;
  delete y;
  delete sum;
  delete mean;
  delete diff;
  delete diff2;
  delete sum_diff2;
  delete mean_diff2;
  delete var;
  delete std;
  delete mul_scale;
  delete new_running_mean;
  delete new_running_var;
}

}  // namespace
}  // namespace frontend
}  // namespace cinn

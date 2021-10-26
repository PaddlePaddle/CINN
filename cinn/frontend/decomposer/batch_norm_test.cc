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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
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

  std::vector<std::pair<std::string, std::vector<float>>> outputs = {
      {"batch_norm_train_moving_mean", new_moving_mean},
      {"batch_norm_train_moving_variance", new_moving_variance},
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

template <typename T>
void cpu_batch_norm_grad(const std::vector<T>& x,
                         const std::vector<T>& dy,
                         const std::vector<T>& scale,
                         const std::vector<T>& save_mean,
                         const std::vector<T>& save_variance,
                         const int n,
                         const int c,
                         const int h,
                         const int w,
                         std::vector<T>* dx,
                         std::vector<T>* dscale,
                         std::vector<T>* dbias,
                         std::vector<T>* grad_std_norm,
                         std::vector<T>* grad_diff,
                         std::vector<T>* grad_std_variance_2d,
                         std::vector<T>* grad_variance_2d_without_mul,
                         std::vector<T>* grad_x0,
                         std::vector<T>* minus_grad_mean) {
  std::vector<T> save_std_varance(c);
  for (int idx = 0; idx < c; ++idx) {
    save_std_varance[idx] = sqrt(save_variance[idx] + 1e-6);
  }
  // grad bias
  memset(dbias->data(), 0, sizeof(float) * c);
  auto func_dbias = [=](int idx, int idy, int idz, int ida) {
    dbias->at(idy) += dy[idx * c * h * w + idy * h * w + idz * w + ida];
  };
  loop(func_dbias, n, c, h, w);

  // grad scale
  memset(dscale->data(), 0, sizeof(float) * c);
  auto func_dscale = [=](int idx, int idy, int idz, int ida) {
    dscale->at(idy) += dy[idx * c * h * w + idy * h * w + idz * w + ida] *
                       ((x[idx * c * h * w + idy * h * w + idz * w + ida] - save_mean[idy]) / save_std_varance[idy]);
  };
  loop(func_dscale, n, c, h, w);

  // grad_std
  auto func_grad_std_norm = [=](int idx, int idy, int idz, int ida) {
    grad_std_norm->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        dy[idx * c * h * w + idy * h * w + idz * w + ida] * scale[idy];
  };
  loop(func_grad_std_norm, n, c, h, w);

  auto func_grad_diff = [=](int idx, int idy, int idz, int ida) {
    grad_diff->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        grad_std_norm->at(idx * c * h * w + idy * h * w + idz * w + ida) / save_std_varance[idy];
  };
  loop(func_grad_diff, n, c, h, w);

  memset(grad_std_variance_2d->data(), 0, sizeof(float) * c);
  auto func_grad_std_variance_2d = [=](int idx, int idy, int idz, int ida) {
    grad_std_variance_2d->at(idy) += -1 * grad_std_norm->at(idx * c * h * w + idy * h * w + idz * w + ida) *
                                     (x[idx * c * h * w + idy * h * w + idz * w + ida] - save_mean[idy]) /
                                     (save_variance[idy] + 1e-6);
  };
  loop(func_grad_std_variance_2d, n, c, h, w);

  for (int idx = 0; idx < c; ++idx) {
    grad_variance_2d_without_mul->at(idx) = grad_std_variance_2d->at(idx) / save_std_varance[idx];
  }

  auto func_grad_x0 = [=](int idx, int idy, int idz, int ida) {
    grad_x0->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        x[idx * c * h * w + idy * h * w + idz * w + ida] * grad_variance_2d_without_mul->at(idy) / (n * h * w);
  };
  loop(func_grad_x0, n, c, h, w);

  memset(minus_grad_mean->data(), 0, sizeof(float) * c);
  auto func_minus_grad_mean = [=](int idx, int idy, int idz, int ida) {
    minus_grad_mean->at(idy) += grad_diff->at(idx * c * h * w + idy * h * w + idz * w + ida);
  };
  loop(func_minus_grad_mean, n, c, h, w);

  for (int idx = 0; idx < c; ++idx) {
    minus_grad_mean->at(idx) += grad_variance_2d_without_mul->at(idx) * save_mean.at(idx);
    minus_grad_mean->at(idx) /= (n * h * w);
  }

  auto func_grad_x = [=](int idx, int idy, int idz, int ida) {
    dx->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        grad_diff->at(idx * c * h * w + idy * h * w + idz * w + ida) +
        grad_x0->at(idx * c * h * w + idy * h * w + idz * w + ida) - minus_grad_mean->at(idy);
  };
  loop(func_grad_x, n, c, h, w);
}

TEST(nn, BATCH_NORM_GRAD) {
  // parameter
  int n = 8, c = 16, h = 4, w = 4;
  int num = n * c * h * w;
  NetBuilder net_builder("net_builder_batch_norm_grad");
  {
    // create input
    auto x             = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto dy            = net_builder.CreateInput(Float(32), {n, c, h, w}, "dy");
    auto scale         = net_builder.CreateInput(Float(32), {c}, "scale");
    auto save_mean     = net_builder.CreateInput(Float(32), {c}, "save_mean");
    auto save_variance = net_builder.CreateInput(Float(32), {c}, "save_variance");

    // add batch norm train
    auto outputs = net_builder.batch_norm_grad(x, dy, scale, save_mean, save_variance);
  }
  // build program
  auto program = net_builder.Build();

  auto target = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto nodes = std::get<0>(graph->topological_order());
  LOG(INFO) << graph->Visualize();

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto run_program = gc.Build();

  // set input
  std::vector<float> x(num), dy(num), scale(c), save_mean(c), save_variance(c);
  InitRandomVector(&x, num);
  InitRandomVector(&dy, num);
  InitRandomVector(&scale, c);
  InitRandomVector(&save_mean, c);
  InitRandomVector(&save_variance, c);
  for (int idx = 0; idx < c; ++idx) {
    save_variance[idx] = abs(save_variance[idx]);
  }

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {
      {"x", x}, {"dy", dy}, {"scale", scale}, {"save_mean", save_mean}, {"save_variance", save_variance}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();

  std::vector<float> dx(num), dscale(c), dbias(c);
  std::vector<float> grad_std_norm(num), grad_diff(num), grad_std_variance_2d(c), grad_variance_2d_without_mul(c),
      grad_x0(num), minus_grad_mean(c);

  cpu_batch_norm_grad(x,
                      dy,
                      scale,
                      save_mean,
                      save_variance,
                      n,
                      c,
                      h,
                      w,
                      &dx,
                      &dscale,
                      &dbias,
                      &grad_std_norm,
                      &grad_diff,
                      &grad_std_variance_2d,
                      &grad_variance_2d_without_mul,
                      &grad_x0,
                      &minus_grad_mean);

  std::vector<std::pair<std::string, std::vector<float>>> outputs = {
      {"batch_norm_grad_bias", dbias},
      {"batch_norm_grad_scale", dscale},
      {"batch_norm_grad_x", dx},
  };

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

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
void random(T* value, int num) {
  for (int idx = 0; idx < num; ++idx) {
    *value++ = rand() / 100000.0f;
  }
}

template <typename T>
void batch_norm_grad(const std::vector<T>& x,
                     const std::vector<T>& dy,
                     const std::vector<T>& scale,
                     const std::vector<T>& save_mean,
                     const std::vector<T>& save_var,
                     const int n,
                     const int c,
                     const int h,
                     const int w,
                     std::vector<T>* dstd,
                     std::vector<T>* ddiff_0,
                     std::vector<T>* dvar,
                     std::vector<T>* ddiff2,
                     std::vector<T>* ddiff_1,
                     std::vector<T>* ddiff,
                     std::vector<T>* dmean,
                     std::vector<T>* dx,
                     std::vector<T>* dscale,
                     std::vector<T>* dbias) {
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
                       ((x[idx * c * h * w + idy * h * w + idz * w + ida] - save_mean[idy]) / save_var[idy]);
  };
  loop(func_dscale, n, c, h, w);

  // grad_std
  auto func_dstd = [=](int idx, int idy, int idz, int ida) {
    dstd->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        dy[idx * c * h * w + idy * h * w + idz * w + ida] * scale[idy];
  };
  loop(func_dstd, n, c, h, w);

  // grad_diff
  auto func_diff_0 = [=](int idx, int idy, int idz, int ida) {
    ddiff_0->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        dstd->at(idx * c * h * w + idy * h * w + idz * w + ida) / save_var[idy];
  };
  loop(func_diff_0, n, c, h, w);

  // grad_var
  memset(dvar, 0, sizeof(float) * c);
  auto func_dvar = [=](int idx, int idy, int idz, int ida) {
    dvar->at(idy) += -1 * dstd->at(idx * c * h * w + idy * h * w + idz * w + ida) / (save_var[idy] * save_var[idy]) *
                     (x[idx * c * h * w + idy * h * w + idz * w + ida] - save_mean[idy]);
  };
  loop(func_dvar, n, c, h, w);

  // grad diff2
  for (int idx = 0; idx < c; ++idx) {
    ddiff2->at(idx) = dvar->at(idx) / (save_var[idx] * float(n * h * w));
  }

  // grad diff2
  auto func_ddiff_1 = [=](int idx, int idy, int idz, int ida) {
    ddiff_1->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        ddiff2->at(idy) * (x[idx * c * h * w + idy * h * w + idz * w + ida] - save_mean[idy]);
  };
  loop(func_ddiff_1, n, c, h, w);

  auto func_diff = [=](int idx, int idy, int idz, int ida) {
    ddiff->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        ddiff_0->at(idx * c * h * w + idy * h * w + idz * w + ida) +
        ddiff_1->at(idx * c * h * w + idy * h * w + idz * w + ida);
  };
  loop(func_diff, n, c, h, w);

  memset(dmean->data(), 0, sizeof(float) * c);
  auto func_dmean = [=](int idx, int idy, int idz, int ida) {
    dmean->at(idy) += -1 * ddiff->at(idx * c * h * w + idy * h * w + idz * w + ida) / float(n * h * w);
  };
  loop(func_dmean, n, c, h, w);

  auto func_dx = [=](int idx, int idy, int idz, int ida) {
    dx->at(idx * c * h * w + idy * h * w + idz * w + ida) =
        ddiff->at(idx * c * h * w + idy * h * w + idz * w + ida) + dmean->at(idy);
  };
  loop(func_dx, n, c, h, w);
}

TEST(nn, BATCH_NORM_GRAD) {
  // parameter
  int n = 8, c = 16, h = 4, w = 4;
  int num = n * c * h * w;
  NetBuilder net_builder("net_builder_batch_norm_grad");
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

  auto target = GetTarget();
  CinnBuilder cinn_builder("cinn_builder_batch_norm_grad");
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
  // hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto run_program = gc.Build();

  // set input
  std::vector<float> x(num), dy(num), scale(c), save_mean(c), save_var(c);
  InitRandomVector(x, num);
  InitRandomVector(dy, num);
  InitRandomVector(scale, c);
  InitRandomVector(save_mean, c);
  InitRandomVector(save_var, c);

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {
      {"x", x}, {"dy", dy}, {"scale", scale}, {"save_mean", save_mean}, {"save_var", save_var}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    // auto* data  = tensor->mutable_data<float>(target);
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();

  std::vector<float> dx(num), dscale(c), dbias(c);
  std::vector<float> dstd(num), ddiff_0(num), dvar(c), ddiff2(c), ddiff_1(num), ddiff(num), dmean(c);
  batch_norm_grad(x,
                  dy,
                  scale,
                  save_mean,
                  save_var,
                  n,
                  c,
                  h,
                  w,
                  &dstd,
                  &ddiff_0,
                  &dvar,
                  &ddiff2,
                  &ddiff_1,
                  &ddiff,
                  &dmean,
                  &dx,
                  &dscale,
                  &dbias);

  std::vector<std::pair<std::string, std::vector<float>>> outputs = {{"var_12", dbias},
                                                                     {"var_18", dscale},
                                                                     /*{"var_20", dstd},
                                                                     {"var_21", ddiff_0},
                                                                     {"var_27", dvar},
                                                                     {"var_31", ddiff2},
                                                                     {"var_33", ddiff_1},
                                                                     {"var_34", ddiff},
                                                                     {"var_38", dmean},*/
                                                                     {"var_40", dx}};

  for (auto& output : outputs) {
    auto tensor = scope->GetTensor(output.first);
    std::vector<float> data(tensor->shape().numel());
    CopyToVector(tensor, &data);
    LOG(INFO) << output.first << " " << tensor->shape().numel();
    for (int idx = 0; idx < tensor->shape().numel(); ++idx) {
      ASSERT_LT(abs((data[idx] - output.second[idx]) / data[idx]), 1e-4);
      // ASSERT_FLOAT_EQ(data[idx], output.second[idx]);
    }
  }
}

}  // namespace
}  // namespace frontend
}  // namespace cinn

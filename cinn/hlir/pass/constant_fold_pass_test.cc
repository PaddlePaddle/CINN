// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace frontend {

TEST(Constant_Folding, fold_broadcast_to_const_scalar_1) {
  NetBuilder net_builder("fold_broadcast_to_const_scalar_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.Constant<float>(1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 5);
}

TEST(Constant_Folding, fold_broadcast_to_const_scalar_2) {
  NetBuilder net_builder("fold_broadcast_to_const_scalar_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.Constant<float>(1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.CreateInput(Float(32), {1}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 10);
}

TEST(Constant_Folding, fold_broadcast_to_const_scalar_3) {
  NetBuilder net_builder("fold_broadcast_to_const_scalar_3");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.Constant<float>(1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.BroadcastTo(A, {h, w, w}, {2});
  auto E = net_builder.CreateInput(Float(32), {h, w, w}, "E");
  auto F = net_builder.Add(B, C);
  auto G = net_builder.Add(D, E);

  auto fetch_ids = {G->id, F->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 10);
}

TEST(Constant_Folding, fold_broadcast_to_fill_constant_1) {
  NetBuilder net_builder("fold_broadcast_to_fill_constant_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({w}, 1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 5);
}

TEST(Constant_Folding, fold_broadcast_to_fill_constant_2) {
  NetBuilder net_builder("fold_broadcast_to_fill_constant_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({w}, 1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.CreateInput(Float(32), {w}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 10);
}

TEST(Constant_Folding, fold_reshape_fill_constant_1) {
  NetBuilder net_builder("fold_reshape_fill_constant_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h * w}, 1.0f, "A");
  auto B = net_builder.Reshape(A, {h, w});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 5);
}

TEST(Constant_Folding, fold_reshape_fill_constant_2) {
  NetBuilder net_builder("fold_reshape_fill_constant_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h * w}, 1.0f, "A");
  auto B = net_builder.Reshape(A, {h, w});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.CreateInput(Float(32), {h * w}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 10);
}

TEST(Constant_Folding, fold_squeeze_fill_constant_1) {
  NetBuilder net_builder("fold_squeeze_fill_constant_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, 1, w, 1}, 1.0f, "A");
  auto B = net_builder.Squeeze(A, {1, 3});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 5);
}

TEST(Constant_Folding, fold_squeeze_fill_constant_2) {
  NetBuilder net_builder("fold_squeeze_fill_constant_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, 1, w, 1}, 1.0f, "A");
  auto B = net_builder.Squeeze(A, {1, 3});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.CreateInput(Float(32), {h, 1, w, 1}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 10);
}

TEST(Constant_Folding, fold_expand_dims_to_fill_constant_1) {
  NetBuilder net_builder("fold_expand_dims_to_fill_constant_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, w}, 1.0f, "A");
  auto B = net_builder.ExpandDims(A, {1, 3});
  auto C = net_builder.CreateInput(Float(32), {h, 1, w, 1}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 5);
}

TEST(Constant_Folding, fold_expand_dims_to_fill_constant_2) {
  NetBuilder net_builder("fold_expand_dims_to_fill_constant_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, w}, 1.0f, "A");
  auto B = net_builder.ExpandDims(A, {1, 3});
  auto C = net_builder.CreateInput(Float(32), {h, 1, w, 1}, "C");
  auto D = net_builder.CreateInput(Float(32),
                                   {
                                       h,
                                       w,
                                   },
                                   "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto program   = net_builder.Build();
  auto target    = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPass(graph.get(), "ConstantFold");
  CHECK_EQ(graph->nodes().size(), 10);
}

}  // namespace frontend
}  // namespace cinn

// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace frontend {

TEST(OpFusionPass, ElementWise_Fusion_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, D);
    auto G = net_builder.ElementwiseAdd(E, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, ElementWise_Fusion_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(E, C);
    auto G = net_builder.ElementwiseAdd(E, D);
    auto H = net_builder.ElementwiseAdd(F, G);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Brodcast_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Brodcast_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(C, A, 0);
    auto F = net_builder.ElementwiseAdd(D, B, 0);
    auto G = net_builder.ElementwiseAdd(E, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Brodcast_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Brodcast_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, D);
    auto G = net_builder.ElementwiseAdd(F, E, 0);
    auto H = net_builder.ElementwiseAdd(G, C);
    auto I = net_builder.ElementwiseAdd(H, D);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(OpFusionPass, Reduce_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, D);
    auto G = net_builder.ReduceSum(F, {0});
    auto H = net_builder.ElementwiseAdd(E, G);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(OpFusionPass, Reduce_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ReduceSum(C, {0});
    auto G = net_builder.ReduceSum(D, {0});
    auto H = net_builder.ElementwiseAdd(E, F);
    auto I = net_builder.ElementwiseAdd(G, H);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Reduce_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ReduceSum(C, {0});
    auto F = net_builder.ReduceSum(D, {1});
    auto G = net_builder.ElementwiseAdd(A, E);
    auto H = net_builder.ElementwiseAdd(B, F);
    auto I = net_builder.ElementwiseAdd(G, H);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(OpFusionPass, Injective_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Injective_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h * 2, w}, "D");

    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.Concat({C, E}, 0);
    auto G = net_builder.ElementwiseAdd(D, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Test_Insert_BroadcastTo) {
  int h = 32, w = 32;
  NetBuilder net_builder("Test_Insert_BroadcastTo");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");

    auto E = net_builder.ElementwiseAdd(C, A, -1);
    auto F = net_builder.ElementwiseAdd(E, B, -1);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  CHECK_EQ(graph->fusion_groups.size(), 1);
}

}  // namespace frontend
}  // namespace cinn

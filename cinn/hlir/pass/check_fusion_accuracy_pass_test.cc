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

#include <string>
#include <unordered_set>

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

int CountAfterPassNodeSize(hlir::framework::Graph* graph) {
  int node_size = 0, output_size = 0;
  for (auto group : graph->fusion_groups) {
    int group_size = group->CollectNodes().size();
    if (group_size == 1) {
      // CheckFusionAccuracyPass will skip if the group only has one node
      continue;
    }

    node_size += group_size;
    output_size += group->GetOutputNodeDatas().size();
  }

  // CheckFusionAccuracyPass will split each group, and add isclose+all+assert node for each output
  return node_size + output_size * 3;
}

TEST(CheckFusionAccuracyPass, ElementWise_Fusion) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_0");
  std::unordered_set<std::string> fetch_ids;
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(E, C);
    auto G = net_builder.ElementwiseAdd(E, D);

    fetch_ids = {F->id, G->id};
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, ElementWise_Fusion_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, D);
    auto G = net_builder.ElementwiseAdd(E, F);
    auto I = net_builder.ElementwiseAdd(E, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, ElementWise_Fusion_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h, w}, "E");
    auto F = net_builder.CreateInput(Float(32), {h, w}, "F");
    auto G = net_builder.ElementwiseAdd(A, B);
    auto H = net_builder.ElementwiseAdd(C, D);
    auto I = net_builder.ElementwiseAdd(E, G);
    auto J = net_builder.ElementwiseAdd(G, H);
    auto K = net_builder.ElementwiseAdd(H, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, ElementWise_Fusion_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h, w}, "E");
    auto F = net_builder.CreateInput(Float(32), {h, w}, "F");
    auto G = net_builder.ElementwiseAdd(A, B);
    auto H = net_builder.ElementwiseAdd(G, C);
    auto I = net_builder.ElementwiseAdd(G, D);
    auto J = net_builder.ElementwiseAdd(G, E);
    auto K = net_builder.ElementwiseAdd(G, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, ElementWise_Fusion_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h, w}, "E");
    auto F = net_builder.CreateInput(Float(32), {h, w}, "F");
    auto G = net_builder.ElementwiseAdd(A, B);
    auto H = net_builder.ElementwiseAdd(G, C);
    auto I = net_builder.ElementwiseAdd(G, D);
    auto J = net_builder.ElementwiseAdd(I, E);
    auto K = net_builder.ElementwiseAdd(I, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, ElementWise_Fusion_5) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.ElementwiseAdd(A, B);
    auto D = net_builder.ElementwiseAdd(A, B);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Broadcast_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, D);
    auto G = net_builder.ElementwiseAdd(F, E);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Broadcast_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, E);
    auto G = net_builder.ElementwiseAdd(D, E);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Broadcast_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, E);
    auto G = net_builder.ElementwiseAdd(D, E);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Broadcast_Test_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h * w, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, E);
    auto G = net_builder.ElementwiseAdd(D, E);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Broadcast_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h, w}, "E");
    auto F = net_builder.ElementwiseAdd(A, B);
    auto G = net_builder.ElementwiseAdd(C, F);
    auto H = net_builder.ElementwiseAdd(D, F);
    auto I = net_builder.ElementwiseAdd(E, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Broadcast_Test_5) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h * w, w}, "E");
    auto F = net_builder.ElementwiseAdd(A, B);
    auto G = net_builder.ElementwiseAdd(C, F);
    auto H = net_builder.ElementwiseAdd(D, F);
    auto I = net_builder.ElementwiseAdd(E, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Reduce_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.ElementwiseAdd(A, B);
    auto D = net_builder.Reduce(C, ReduceKind::kSum, {0});
    auto E = net_builder.Reduce(C, ReduceKind::kSum, {0});
    auto F = net_builder.Reduce(C, ReduceKind::kSum, {0});
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Reduce_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.ElementwiseAdd(A, B);
    auto D = net_builder.Reduce(C, ReduceKind::kSum, {0});
    auto E = net_builder.Reduce(C, ReduceKind::kSum, {1});
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Reduce_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.ElementwiseAdd(A, B);
    auto E = net_builder.Reduce(D, ReduceKind::kSum, {0});
    auto F = net_builder.Reduce(D, ReduceKind::kSum, {1});
    auto G = net_builder.ElementwiseAdd(C, E);
    auto H = net_builder.ElementwiseAdd(C, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Reduce_Test_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.Reduce(E, ReduceKind::kSum, {0});
    auto G = net_builder.ElementwiseAdd(C, F);
    auto H = net_builder.ElementwiseAdd(D, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Reduce_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.Reduce(E, ReduceKind::kSum, {0});
    auto G = net_builder.ElementwiseAdd(C, F);
    auto H = net_builder.ElementwiseAdd(D, F);
    auto I = net_builder.ElementwiseAdd(D, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

TEST(FusionMergePass, Reduce_Test_5) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Test_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.ElementwiseAdd(A, B);
    auto D = net_builder.Reduce(A, ReduceKind::kSum, {1});
    auto E = net_builder.Reduce(B, ReduceKind::kSum, {1});
    auto F = net_builder.Reduce(C, ReduceKind::kSum, {1});
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  int group_size_affter = graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPass(graph.get(), "CheckFusionAccuracyPass");
  VLOG(1) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_affter);
}

}  // namespace cinn::frontend

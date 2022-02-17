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

#include "cinn/common/macros.h"
#include "cinn/hlir/framework/pass.h"

namespace cinn {
namespace hlir {
namespace pass {

bool IsOutOfTranspose(const common::GraphNode* graph_node) {
  for (auto& inlink : graph_node->inlinks()) {
    auto node = inlink->source()->safe_as<framework::Node>();
    return node && "transpose" == node->op()->name;
  }
  return false;
}

bool CanFoldIntoDot(const common::GraphNode* graph_node) {
  auto node = graph_node->safe_as<framework::Node>();
  if (node && "matmul" == node->op()->name) {
    for (auto inlink : node->inlinks_in_order()) {
      if (IsOutOfTranspose(inlink->source())) {
        return true;
      }
    }
  }
  return false;
}

bool CanFoldIntoConv(const common::GraphNode* graph_node) {
  auto node = graph_node->safe_as<framework::Node>();
  if (node && "conv2d" == node->op()->name) {
    for (auto inlink : node->inlinks_in_order()) {
      if (IsOutOfTranspose(inlink->source())) {
        return true;
      }
    }
  }
  return false;
}

void FoldIntoDot(common::GraphNode* conv) { VLOG(1) << "Fold Into Dot"; }

void FoldIntoConv(common::GraphNode* conv) { VLOG(1) << "Fold Into Conv"; }

void TransposeFolding(framework::Graph* graph) {
  // 1. Get dot nodes which can fold transpose into itself.
  // 2. Get conv nodes ...
  // 3. Folding dot with transpose.
  // 4. Folding conv with transpose.
  auto foldable_dots = graph->CollectNodes(CanFoldIntoDot);
  for (auto dot : foldable_dots) {
    FoldIntoDot(dot);
  }
  auto foldable_convs = graph->CollectNodes(CanFoldIntoConv);
  for (auto conv : foldable_convs) {
    FoldIntoConv(conv);
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(TransposeFolding) {
  CINN_REGISTER_PASS(TransposeFolding)
      .describe("This pass folding transpose into dot/conv.")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::TransposeFolding);

  return true;
}

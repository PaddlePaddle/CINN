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
#include <unordered_set>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::GraphNode;
using common::Type;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;
using framework::OpPatternKind;

void ElementwiseReduceFusePass(Graph* graph) {}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(ElementwiseReduceFuse) {
  CINN_REGISTER_PASS(ElementwiseReduceFuse)
      .describe("This pass fuse elementwise and reduce op.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::ElementwiseReduceFusePass);

  return true;
}

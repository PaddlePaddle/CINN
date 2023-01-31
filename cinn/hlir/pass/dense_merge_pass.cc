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

#include "cinn/common/type.h"
#include "cinn/hlir/pass/fusion_merge_base.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;

class DenseMergePassHelper : public FusionHelperBase {
 public:
  DenseMergePassHelper(Graph* graph) : FusionHelperBase(graph), graph_(graph) {}

  void operator()() {}

 private:
  Graph* graph_;
} :

    void
    DenseMergePassInternal(Graph* graph) {
  DenseMergePassHelper dense_merge_pass_helper(graph);
  dense_merge_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(DenseMerge) {
  CINN_REGISTER_PASS(DenseMerge)
      .describe("")
      .set_change_structure(true)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::DenseMergePassInternal);
  return true;
}

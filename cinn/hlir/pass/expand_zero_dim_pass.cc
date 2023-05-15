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

#include "cinn/hlir/pass/fusion_helper_base.h"
#include "glog/logging.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::shape_t;

class ExpandZeroDimPassHelper : public FusionHelperBase {
 public:
  ExpandZeroDimPassHelper(Graph* graph) : FusionHelperBase(graph), graph_(graph) {}

  void operator()() {
    auto& shape_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
    for (auto& iter : shape_dict) {
      if (iter.second.empty()) {
        VLOG(4) << "Change 0D-Tensor " << iter.first << " to 1D-Tensor";
        iter.second.push_back(1);
      }
    }
  }

 private:
  Graph* graph_;
};

void ExpandZeroDimPass(Graph* graph) {
  ExpandZeroDimPassHelper expand_zero_dim_pass_helper(graph);
  expand_zero_dim_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(ExpandZeroDimPass) {
  CINN_REGISTER_PASS(ExpandZeroDimPass)
      .describe(
          "Expand 0D-Tensor to 1D-Tensor. Make sure this is the first pass of graph pass, and it changes the shape of "
          "0D-Tensor (both input and output) to 1 Dim. A better way to support 0D-Tensor is fully supporting 0D-Tensor "
          "in the codegen process, as the codegen module might reconstruct later, we temporarily use pass to support "
          "this mechanism.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::ExpandZeroDimPass);

  return true;
}

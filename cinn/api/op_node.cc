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

#include "cinn/api/op_node.h"

#include "cinn/api/tensor_node.h"

namespace cinn {
namespace api {

TensorNode OpNode::GetInput(size_t i) const {
  auto edges = node_->inlinks_in_order();
  return TensorNode(helper_, edges[i]->safe_as<hlir::framework::NodeData>());
}

TensorNode OpNode::GetOutput(size_t i) const {
  auto edges = node_->outlinks_in_order();
  return TensorNode(helper_, edges[i]->safe_as<hlir::framework::NodeData>());
}

}  // namespace api
}  // namespace cinn
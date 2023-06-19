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

#pragma once

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace api {

class OpNode;

class TensorNode {
 public:
  TensorNode(const hlir::pass::FusionHelperBase* helper, const hlir::framework::NodeData* node_data) : helper_(helper), node_data_(node_data) {}

  // Get the shape of tensor.
  const shpae_t& Shape() const {
    return helper_->GetNodeDataShape(node_data_)
  }

  OpNode Producer() const;

  size_t ConsumerSize() const {
    return node_data_->outlinks().size();
  }

  OpNode Consumer(size_t index) const;

 private:
  const hlir::pass::FusionHelperBase* helper_;
  const hlir::framework::NodeData* node_data_;
};

}  // namespace api
}  // namespace cinn

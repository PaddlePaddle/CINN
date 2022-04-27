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

#include <algorithm>
#include <unordered_set>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using namespace framework;

class FusionHelperBase {
 public:
  FusionHelperBase(const absl::flat_hash_map<std::string, shape_t>& shape_dict, const common::Target target)
      : shape_dict_(shape_dict), target_(target) {
    // get op pattern dict
    op_pattern_dict_ = &framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
  }

 protected:
  OpPatternKind GetOpKind(const framework::Node* node) {
    CHECK(op_pattern_dict_->Find(node->op())) << "Don't find the pattern of op : " << node->id();
    auto kind = op_pattern_dict_[0][node->op()];

    CHECK_NE(kind, framework::kTuple) << "kTuple is not support now!";
    if (kind == framework::kBroadcast) {
      // As binary op was defined as broadcast, actually it should be element-wise.
      if (node->op()->name != "broadcast_to") {
        return framework::kElemWise;
      }
    }

    return kind;
  }

  NodeData* GetNodeData(const Node* node) {
    auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
    CHECK(node_data);
    return node_data;
  }

  shape_t GetNodeDataShape(const Node* node) {
    auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
    CHECK(node_data);
    CHECK(shape_dict_.count(node_data->id())) << "Can't find " << node_data->id() << " 's shape!";
    return shape_dict_.at(node_data->id());
  }

  std::vector<NodeData*> GetProducerNodeData(const Node* node) {
    std::vector<NodeData*> producer_node_data;
    for (auto& edge : node->inlinks()) {
      auto graph_node    = edge->source();
      auto producer_data = graph_node->safe_as<NodeData>();
      CHECK(producer_data);
      producer_node_data.push_back(producer_data);
    }
    return producer_node_data;
  }

  bool WithoutLastDimInReduce(const std::vector<int>& inshape, const std::vector<int>& axes) {
    // if last axis is in reduce.
    if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
        std::find(axes.begin(), axes.end(), -1) != axes.end()) {
      return false;
    }

    int sum_last_axes = 1;
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      sum_last_axes *= inshape[idx];
    }

    if (sum_last_axes > 1) {
      return true;
    } else {
      return false;
    }
  }
  // target
  common::Target target_;
  // shape dict
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;
  // op pattern dict
  const framework::OpValueType<OpPatternKind>* op_pattern_dict_;
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

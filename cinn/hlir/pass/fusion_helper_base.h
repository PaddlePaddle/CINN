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
  FusionHelperBase(const framework::Graph* graph)
      : shape_dict_(graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape")), target_(graph->target_) {
    // get op pattern dict
    op_pattern_dict_ = &framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
    // output node set
    for (auto node_data : graph->outputs) {
      CHECK(node_data->source_node.get());
      output_nodes_set_.insert(node_data->source_node.get());
    }
  }

 protected:
  OpPatternKind GetOpKind(const framework::Node* node) {
    CHECK(op_pattern_dict_->Find(node->op())) << "Don't find the pattern of op : " << node->id();
    auto kind = op_pattern_dict_[0][node->op()];

    if (kind == framework::kBroadcast) {
      // As binary op was defined as broadcast, actually it should be element-wise.
      if (node->op()->name != "broadcast_to") {
        return framework::kElementWise;
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

  int GetSharedSize(const Node* node) {
    auto producers = GetProducerNodeData(node);
    CHECK_GT(producers.size(), 0);
    auto inshape = shape_dict_.at(producers[0]->id());
    auto axes    = absl::get<std::vector<int>>(node->attrs.attr_store.at("dim"));
    if (WithoutLastDimInReduce(inshape, axes)) {
      int lane = 1;
      for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
        lane = inshape[idx];
      }
      int max_num_threads = common::DefaultNVGPUTarget().max_num_threads();
      if (lane > max_num_threads / 2) {
        return 0;
      }
      int index = axes.size() - 1;
      for (; index >= 0; --index) {
        if (index + 1 < axes.size() && axes[index] != axes[index + 1] - 1) {
          break;
        }
        lane *= inshape[axes[index]];
        if (lane > max_num_threads / 2) {
          break;
        }
      }
      // if lane > (max_num_threads / 2),the loop break from lane > max_num_threads / 2.
      int axis = lane > (max_num_threads / 2) ? axes[index] : axes[index + 1];
      if (lane <= max_num_threads) {
        return lane * sizeof(float);
      } else {
        int prefix = inshape[axis];
        int tail   = lane / prefix;
        for (int idx = max_num_threads / tail; idx > ((max_num_threads / 2) / tail); --idx) {
          if (prefix % idx == 0) {
            return idx * tail * sizeof(float);
          }
        }
        int num = max_num_threads / tail;
        return num * tail * sizeof(float);
      }
    }
    return 0;
  }
  // target
  const common::Target& target_;
  // output node set
  std::unordered_set<Node*> output_nodes_set_;
  // shape dict
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;
  // op pattern dict
  const framework::OpValueType<OpPatternKind>* op_pattern_dict_;
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

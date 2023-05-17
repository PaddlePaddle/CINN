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

#include "cinn/hlir/pass/constant_folding_pass_util.h"

#include <algorithm>
#include <cstddef>
#include <queue>

#include "absl/types/variant.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/utils/functional.h"
#include "cinn/utils/type_defs.h"

namespace cinn {
namespace hlir {
namespace pass {
namespace utils {

using cinn::utils::Attribute;
using cinn::utils::AttributeMap;

class ConstantFoldingHelper {
 public:
  ConstantFoldingHelper(const FusionHelperBase* helper, Graph* graph, Node* node)
      : helper_(helper), graph_(graph), consumer_(node), producer_(helper->GetProducerNode(node)[0]) {}

  const AttributeMap& GetProducerAttrs() const { return producer_->attrs.attr_store; }
  const AttributeMap& GetConsumerAttrs() const { return consumer_->attrs.attr_store; }

  // fold consumer node and producer node into a new op node
  void operator()(const AttributeMap& attrs_map, const std::string& new_op_name) {
    auto* new_fold_node = CreateNewNode(new_op_name, attrs_map);

    // create new link.
    RelinkEdge(new_fold_node);
  }

  // fold consumer node into producer node
  void operator()(const AttributeMap& attrs_map) { this->operator()(attrs_map, producer_->op()->name); }

 private:
  Node* CreateNewNode(const std::string& op_name, const AttributeMap& attrs_map) {
    auto* node             = new Node(Operator::Get(op_name), op_name, common::UniqName(op_name));
    node->attrs.attr_store = attrs_map;
    graph_->RegisterNode(node->id(), node);
    return node;
  }

  void RelinkEdge(Node* new_fold_node) {
    // first relink consumer node.
    RelinkAndRemoveConsumer(new_fold_node);
    // then relink producer node.
    RelinkProducer(new_fold_node);
  }

  void RelinkAndRemoveConsumer(Node* new_fold_node) {
    // relink outputs
    {
      const auto& consumer_outputs = helper_->GetNodeDatas(consumer_);
      for (auto* output : consumer_outputs) {
        // now the output linked to new fold node
        output->source_node.Reset(new_fold_node);
        new_fold_node->LinkTo(output);

        consumer_->UnLinkSingleTo(output);
      }
    }

    // consumer are replaced by new_fold_node now, drop useless consumer node
    {
      const auto& consumer_inputs = helper_->GetProducerNodeData(consumer_);
      for (auto* input : consumer_inputs) {
        input->UnLinkSingleTo(consumer_);
      }
      graph_->DropNode(consumer_);
    }
  }

  void RelinkProducer(Node* new_fold_node) {
    // if the producer's output are fetched, cannot remove the producer node
    bool can_producer_remove = !helper_->output_nodes_set_.count(producer_);
    // check whether producer node can be removed
    if (can_producer_remove) {
      const auto& producer_outputs = helper_->GetNodeDatas(producer_);
      for (auto* output : producer_outputs) {
        if (!output->outlinks().empty()) {
          // if the producer's output linked to other node, cannot remove
          can_producer_remove = false;
          break;
        }
      }
    }

    // relink inputs
    {
      const auto& producer_inputs = helper_->GetProducerNodeData(producer_);
      for (auto* input : producer_inputs) {
        input->LinkTo(new_fold_node);

        if (can_producer_remove) {
          input->UnLinkSingleTo(producer_);
        }
      }
    }

    // drop producer node if needed
    if (can_producer_remove) {
      // the producer's output are no need now, remove
      const auto& producer_outputs = helper_->GetNodeDatas(producer_);
      for (auto* output : producer_outputs) {
        producer_->UnLinkSingleTo(output);
        graph_->DropNode(output);
      }

      graph_->DropNode(producer_);
    }
  }

  const FusionHelperBase* helper_;
  Graph* graph_{nullptr};
  Node* producer_{nullptr};
  Node* consumer_{nullptr};
};

}  // namespace utils

inline void fold_broadcast_to_constant(const FusionHelperBase* helper, Graph* graph, Node* node) {
  auto constant_op = helper->GetProducerNode(node)[0];
  CHECK(node->attrs.attr_store.count("out_shape"));
  auto shape = absl::get<std::vector<int>>(node->attrs.attr_store.at("out_shape"));
  CHECK(constant_op->attrs.attr_store.count("value"));
  // create constant op.
  Node* node_tmp = utils::CreateNewNode("fill_constant");
  // set node attr
  node_tmp->attrs.attr_store["dtype"]     = constant_op->attrs.attr_store.at("dtype");
  node_tmp->attrs.attr_store["shape"]     = shape;
  node_tmp->attrs.attr_store["value"]     = constant_op->attrs.attr_store.at("value");
  node_tmp->attrs.attr_store["force_cpu"] = false;
  graph->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.Reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto constant_node_data = helper->GetNodeData(constant_op);
  if (constant_node_data->outlinks().size() == 1) {
    graph->DropNode(node);
    graph->DropNode(constant_op);
    graph->DropNode(constant_node_data);
  } else {
    constant_node_data->UnLinkSingleTo(node);
    graph->DropNode(node);
  }
}

inline void fold_reshape_fill_constant(const FusionHelperBase* helper, Graph* graph, Node* node) {
  auto constant_op = helper->GetProducerNode(node)[0];
  CHECK(node->attrs.attr_store.count("shape"));
  auto shape = absl::get<std::vector<int>>(node->attrs.attr_store.at("shape"));
  CHECK(constant_op->attrs.attr_store.count("value"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"), "fill_constant", common::UniqName("fill_constant"));
  // set node attr
  node_tmp->attrs.attr_store["dtype"]     = constant_op->attrs.attr_store.at("dtype");
  node_tmp->attrs.attr_store["shape"]     = shape;
  node_tmp->attrs.attr_store["value"]     = constant_op->attrs.attr_store.at("value");
  node_tmp->attrs.attr_store["force_cpu"] = false;
  graph->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.Reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto constant_node_data = helper->GetNodeData(constant_op);
  if (constant_node_data->outlinks().size() == 1) {
    graph->DropNode(node);
    graph->DropNode(constant_op);
    graph->DropNode(constant_node_data);
  } else {
    constant_node_data->UnLinkSingleTo(node);
    graph->DropNode(node);
  }
}

// fold fill_constant->squeeze ==> fill_constant
inline void fold_squeeze_fill_constant(const FusionHelperBase* helper, Graph* graph, Node* node) {
  auto constant_op = helper->GetProducerNode(node)[0];
  CHECK(constant_op->attrs.attr_store.count("shape"));
  auto shape = absl::get<std::vector<int>>(constant_op->attrs.attr_store.at("shape"));
  CHECK(node->attrs.attr_store.count("axes"));
  auto axes = absl::get<std::vector<int>>(node->attrs.attr_store.at("axes"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"), "fill_constant", common::UniqName("fill_constant"));
  // set node attr
  std::vector<int> n_shape;
  if (axes.size() == 0) {
    for (auto s : shape) {
      if (s > 1) {
        n_shape.push_back(s);
      }
    }
  } else {
    for (int idx = 0; idx < shape.size(); ++idx) {
      if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
        n_shape.push_back(shape[idx]);
      }
    }
  }

  node_tmp->attrs.attr_store["dtype"]     = constant_op->attrs.attr_store.at("dtype");
  node_tmp->attrs.attr_store["shape"]     = n_shape;
  node_tmp->attrs.attr_store["value"]     = constant_op->attrs.attr_store.at("value");
  node_tmp->attrs.attr_store["force_cpu"] = false;
  graph->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.Reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto constant_node_data = helper->GetNodeData(constant_op);
  if (constant_node_data->outlinks().size() == 1) {
    graph->DropNode(node);
    graph->DropNode(constant_op);
    graph->DropNode(constant_node_data);
  } else {
    constant_node_data->UnLinkSingleTo(node);
    graph->DropNode(node);
  }
}

// fold fill_constant->expand_dims ==> fill_constant
inline void fold_expand_dims_fill_constant(const FusionHelperBase* helper, Graph* graph, Node* node) {
  auto constant_op = helper->GetProducerNode(node)[0];
  CHECK(constant_op->attrs.attr_store.count("shape"));
  auto shape = absl::get<std::vector<int>>(constant_op->attrs.attr_store.at("shape"));
  CHECK(node->attrs.attr_store.count("axes"));
  auto axes = absl::get<std::vector<int>>(node->attrs.attr_store.at("axes"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"), "fill_constant", common::UniqName("fill_constant"));
  int shape_size = shape.size();
  int axes_size  = axes.size();
  int total_size = shape_size + axes_size;
  // check axes whether in range [-total_size, total_size-1] and convert all to [0, total_size-1].
  axes = utils::GetPositiveAxes(axes, total_size);
  // check axes can't repeat.
  std::sort(axes.begin(), axes.end(), std::less<int>());
  for (int idx = 0; idx < axes_size - 1; ++idx) {
    CHECK_NE(axes[idx], axes[idx + 1]);
  }
  // insert 1 to new shape.
  std::vector<int> n_shape(total_size, 1);
  for (int idx = 0, index = 0; idx < n_shape.size(); ++idx) {
    if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
      n_shape[idx] = shape[index++];
    }
  }

  // set node attr
  node_tmp->attrs.attr_store["dtype"]     = constant_op->attrs.attr_store.at("dtype");
  node_tmp->attrs.attr_store["shape"]     = n_shape;
  node_tmp->attrs.attr_store["value"]     = constant_op->attrs.attr_store.at("value");
  node_tmp->attrs.attr_store["force_cpu"] = false;
  graph->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.Reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto constant_node_data = helper->GetNodeData(constant_op);
  if (constant_node_data->outlinks().size() == 1) {
    graph->DropNode(node);
    graph->DropNode(constant_op);
    graph->DropNode(constant_node_data);
  } else {
    constant_node_data->UnLinkSingleTo(node);
    graph->DropNode(node);
  }
}

// fold reshape->broadcast_to ==> broadcast_to
inline void fold_broadcast_to_reshape(const FusionHelperBase* helper, Graph* graph, Node* node) {}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

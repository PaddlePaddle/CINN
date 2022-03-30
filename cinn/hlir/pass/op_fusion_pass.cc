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

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::OpPatternKind;
using framework::shape_t;

using common::GraphEdge;
using common::GraphNode;

using Group  = std::shared_ptr<Graph::Group>;
using Groups = std::vector<Group>;

using ShapeDict         = absl::flat_hash_map<std::string, shape_t>;
using ConditionFunction = std::function<bool(const Node*, const Node*)>;

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class OpFusionPassHelper {
 public:
  OpFusionPassHelper(std::vector<GraphNode*>& graph_nodes, const absl::flat_hash_map<std::string, shape_t>& shape_dict)
      : shape_dict_(shape_dict) {
    // init fusion relation
    InitFusionRelation();
    // get op pattern dict
    op_pattern_dict_ = &framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
    // filter node data, create group for each node
    for (auto graph_node : graph_nodes) {
      auto node = graph_node->safe_as<Node>();
      if (node) {
        nodes_.push_back(node);
        auto group = std::make_shared<Graph::Group>();
        // init group
        group->nodes.push_back(node);
        group->nodes_set.insert(node);
        group->output_nodes.insert(node);
        // input node
        for (auto& edge : node->inlinks()) {
          auto input_graph_node = edge->source();
          auto input_node_data  = input_graph_node->safe_as<NodeData>();
          CHECK(input_node_data);
          // input data has noe source node
          if (input_node_data->source_node.get()) {
            group->input_nodes.insert(input_node_data->source_node.get());
          }
        }

        // group type
        group->op_pattern_kind = GetOpKind(node);
        // use current node as master node for schedule
        group->master_nodes.insert(node);
        fusion_groups_[node] = group;
      }
    }
    // reverse node for output to input
    std::reverse(nodes_.begin(), nodes_.end());
  }

  // return a vector of groups in topological order.
  Groups operator()() {
    // do op fusion.
    DoOpFusion();

    // find all fusion group.
    Groups fusion_groups;
    std::unordered_set<Graph::Group*> groups_set;
    for (auto node : nodes_) {
      auto& group = fusion_groups_[node];
      if (!groups_set.count(group.get())) {
        groups_set.insert(group.get());
        fusion_groups.push_back(group);
      }
    }

    for (auto& group : fusion_groups) {
      // find producer group
      for (auto& node : group->input_nodes) {
        for (auto& edge : node->inlinks()) {
          auto producer_node      = edge->source();
          auto producer_node_data = producer_node->safe_as<NodeData>();
          CHECK(producer_node_data);
          // input data has no source node
          if (producer_node_data->source_node.get()) {
            auto producer_group = fusion_groups_[producer_node_data->source_node.get()];
            group->producer_groups.insert(producer_group.get());
          }
        }
      }
      // find consumer group
      auto output_node      = *group->output_nodes.begin();
      auto output_node_data = (*output_node->outlinks().begin())->sink();
      for (auto& link : output_node_data->outlinks()) {
        auto consumer_node = link->sink()->safe_as<Node>();
        CHECK(consumer_node);
        auto consumer_group = fusion_groups_[consumer_node];
        group->consumer_groups.insert(consumer_group.get());
      }
    }

    // reverse to keep fusion group in order.
    std::reverse(fusion_groups.begin(), fusion_groups.end());
    return fusion_groups;
  }

 private:
  OpPatternKind GetOpKind(const Node* node) {
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

  void DoOpFusion() {
    for (auto consumer : nodes_) {
      // kOpaque op can't fuse any other op.
      if (GetOpKind(consumer) == framework::kOpaque) {
        continue;
      }

      // As others + kBroacast is left to 'Fusion Merge Pass'.
      if (GetOpKind(consumer) == framework::kBroadcast) {
        continue;
      }

      // fusion op for consumer
      auto consumer_fusion = fusion_groups_[consumer];
      // check all linkin node
      for (auto& edge : consumer->inlinks()) {
        auto graph_node    = edge->source();
        auto producer_data = graph_node->safe_as<NodeData>();
        CHECK(producer_data);

        auto producer = producer_data->source_node.get();
        // if producer is fused.
        if (consumer_fusion->nodes_set.count(producer)) {
          VLOG(11) << "Op " << producer->id() << " is fused.";
          continue;
        }
        // if producer data is placeholder
        if (!producer) {
          continue;
        }

        // kOpaque op can't fuse any other op.
        if (GetOpKind(producer) == framework::kOpaque) {
          continue;
        }
        bool can_fuse = true;
        // checkout producer node outputs are all in fusion op
        for (auto& link : producer_data->outlinks()) {
          auto consumer_node = link->sink()->safe_as<Node>();
          CHECK(consumer_node);
          // if fusion group can't find node, can't merge
          if (consumer_fusion->nodes_set.find(consumer_node) == consumer_fusion->nodes_set.end()) {
            can_fuse = false;
            break;
          }
        }

        if (!can_fuse || !CanFuse(producer, consumer)) continue;
        VLOG(11) << "Fuse Op " << producer->id() << " into Op " << consumer->id();

        // fuse producer to fusion group
        if (consumer_fusion->nodes_set.find(producer) == consumer_fusion->nodes_set.end()) {
          consumer_fusion->nodes.push_back(producer);
        }
        consumer_fusion->nodes_set.insert(producer);
        consumer_fusion->input_nodes.erase(producer);
        consumer_fusion->op_pattern_kind =
            static_cast<int>(consumer_fusion->op_pattern_kind) > static_cast<int>(GetOpKind(producer))
                ? consumer_fusion->op_pattern_kind
                : GetOpKind(producer);

        if (GetOpKind(producer) == framework::kCommReduce) {
          consumer_fusion->master_nodes.insert(producer);
        }

        if (producer_data->outlinks().size() > 1) {
          consumer_fusion->internal_nodes.insert(producer);
        }

        // add input node
        for (auto& edge : producer->inlinks_in_order()) {
          auto input_node      = edge->source();
          auto input_node_data = input_node->safe_as<NodeData>();
          CHECK(input_node_data);
          if (input_node_data->source_node.get()) {
            consumer_fusion->input_nodes.insert(input_node_data->source_node.get());
          }
        }

        // update node group
        fusion_groups_[producer] = consumer_fusion;
      }
    }
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

  void InitFusionRelation() {
    // fusion op fuse condition function.
    // 1. always can fuse.
    auto always_fuse = [](const Node* producer, const Node* consumer) -> bool { return true; };
    // 2. cant fuse.
    auto no_fuse = [](const Node* producer, const Node* consumer) -> bool { return false; };
    // 3. has same output shape.
    auto is_same_shape = [this](const Node* producer, const Node* consumer) -> bool {
      auto& fusion_op  = this->fusion_groups_[consumer];
      auto master_node = fusion_op->master_nodes.begin();
      return this->GetNodeDataShape(producer) == this->GetNodeDataShape(*master_node) ? true : false;
    };
    // 4. without last dimension in reduce axis.
    auto without_last_dimension_in_reduce = [this](const Node* producer, const Node* consumer) -> bool {
      auto reduce_dim = absl::get<std::vector<int>>(producer->attrs.attr_store.at("dim"));
      auto shape      = this->shape_dict_.at(producer->inlinks_in_order()[0]->source()->id());
      // check last dimension in reduce.
      if (std::find(reduce_dim.begin(), reduce_dim.end(), shape.size() - 1) == reduce_dim.end() &&
          std::find(reduce_dim.begin(), reduce_dim.end(), -1) == reduce_dim.end()) {
        return true;
      }
      return false;
    };
    // 5. checkout reduce op has same attr.
    auto reduce_fuse_reduce = [this](const Node* producer, const Node* consumer) -> bool {
      auto& fusion_op = this->fusion_groups_[consumer];
      Node* reducer   = NULL;
      for (auto* master : fusion_op->master_nodes) {
        if (this->GetOpKind(master) == framework::kCommReduce) {
          reducer = master;
          break;
        }
      }
      // check reduce has same input shape and output shape
      auto producer_input_shape  = shape_dict_.at(producer->inlinks_in_order()[0]->source()->id());
      auto producer_output_shape = shape_dict_.at(producer->outlinks_in_order()[0]->sink()->id());

      auto reducer_input_shape  = shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());
      auto reducer_output_shape = shape_dict_.at(reducer->outlinks_in_order()[0]->sink()->id());

      auto producer_reduce_dim = absl::get<std::vector<int>>(producer->attrs.attr_store.at("dim"));
      auto reducer_reduce_dim  = absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));

      for (auto& dim : producer_reduce_dim) {
        // if dim = -1, set as shape.size() - 1
        if (dim == -1) {
          dim = producer_input_shape.size() - 1;
        }
      }

      for (auto& dim : reducer_reduce_dim) {
        // if dim = -1,  set as shape.size() - 1
        if (dim == -1) {
          dim = reducer_input_shape.size() - 1;
        }
      }
      // check shape is same
      if (producer_input_shape != reducer_input_shape || producer_output_shape != reducer_output_shape ||
          producer_reduce_dim != reducer_reduce_dim ||
          absl::get<std::vector<int>>(producer->attrs.attr_store.at("keep_dim")) !=
              absl::get<std::vector<int>>(reducer->attrs.attr_store.at("keep_dim"))) {
        return false;
      }

      return true;
    };
    // fusion relation.
    // 1.kElementwise as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElemWise, framework::kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Elementwise + *Elementwise*). As has same output shape, can always fuse.
          {framework::kElemWise, always_fuse},
          // must be horizontal, as Elementwise + Broadcast is left to fusion merge pass.
          {framework::kBroadcast, is_same_shape},
          // must be horizontal, as Elementwise + Reduce is left to fusion merge pass.
          {framework::kCommReduce, is_same_shape},
          // can be horizontal or can compute inline, check with same output shape or can compute inline.
          {framework::kInjective,
           [this, is_same_shape](const Node* producer, const Node* consumer) -> bool {
             return is_same_shape(producer, consumer) || this->GetNodeData(producer)->outlinks().size() == 1;
           }},
          // must be horizontal, check with same output shape.
          {framework::kOutEWiseFusable, is_same_shape}};
      fusion_relation_map_[framework::kElemWise] = std::move(relation);
    }
    // 2.kBroadcast as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElemWise, framework::kCommReduce, framework::kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Broadcast + *Elementwise*), check with same output shape.
          {framework::kElemWise, is_same_shape},
          // must be horizontal, as Broadcast + Broadcast is not allowed.
          {framework::kBroadcast, is_same_shape},
          // horizontal or vertical relation(Broadcast + Reduce).
          {framework::kCommReduce, always_fuse},
          // can be horizontal or can compute inline, check with same output shape or just one consumer.
          {framework::kInjective,
           [this, is_same_shape](const Node* producer, const Node* consumer) -> bool {
             return is_same_shape(producer, consumer) || this->GetNodeData(producer)->outlinks().size() == 1;
           }},
          // must be horizontal, check with same output shape.
          {framework::kOutEWiseFusable, is_same_shape}};
      fusion_relation_map_[framework::kBroadcast] = std::move(relation);
    }
    // 3.kCommReduce as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElemWise};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Reduce + Elementwise*), check without last dimension in reduce.
          {framework::kElemWise, without_last_dimension_in_reduce},
          // must be horizontal relation, check with same output shape and without last dimension in reduce.
          {framework::kBroadcast,
           [this, is_same_shape, without_last_dimension_in_reduce](const Node* producer, const Node* consumer) -> bool {
             return is_same_shape(producer, consumer) && without_last_dimension_in_reduce(producer, consumer);
           }},
          // must be horizontal relation and with same reduce attr.
          {framework::kCommReduce, reduce_fuse_reduce},
          // can't fuse.
          {framework::kInjective, no_fuse},
          // can't fuse.
          {framework::kOutEWiseFusable, no_fuse}};
      fusion_relation_map_[framework::kCommReduce] = std::move(relation);
    }
    // 4.kInjective
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElemWise, framework::kCommReduce};
      // producer -> fusion
      relation.fusion_op_kind = {
          // can be horizontal or vertical(Injective + Elementwise), check with same output shape.
          {framework::kElemWise, is_same_shape},
          // must be horizontal relation, check with same output shape.
          {framework::kBroadcast, is_same_shape},
          // left to fusion merge pass.
          {framework::kCommReduce, no_fuse},
          // must be horizontal relation, check with same output shape.
          {framework::kInjective, is_same_shape},
          // can't fuse.
          {framework::kOutEWiseFusable, no_fuse},
      };
      fusion_relation_map_[framework::kInjective] = std::move(relation);
    }
    // 5.kOutEWiseFusable
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {framework::kElemWise};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation, check has same shape.
          {framework::kElemWise, is_same_shape},
          // it must be horizontal relation, check has same shape.
          {framework::kBroadcast, is_same_shape},
          // can't fuse.
          {framework::kCommReduce, no_fuse},
          // must be horizontal relation, check has same shape.
          {framework::kInjective, is_same_shape},
          // can't fuse.
          {framework::kOutEWiseFusable, no_fuse},
      };
      fusion_relation_map_[framework::kOutEWiseFusable] = std::move(relation);
    }
  }

  bool CanFuse(const Node* producer, const Node* consumer) {
    auto& relation = fusion_relation_map_[GetOpKind(producer)];
    if (relation.op_kind.count(GetOpKind(consumer))) {
      return relation.fusion_op_kind[GetOpKind(consumer)](producer, consumer);
    }

    return false;
  }

  std::vector<Node*> nodes_;
  std::unordered_map<const Node*, Group> fusion_groups_;
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;
  // a node map
  std::unordered_set<Node*> nodes_set_;
  // op pattern dict
  const framework::OpValueType<OpPatternKind>* op_pattern_dict_;

  struct FusionRelation {
    // producer -> consumer
    std::unordered_set<framework::OpPatternKind> op_kind = {};
    // producer -> fusion sonsumer
    std::unordered_map<framework::OpPatternKind, ConditionFunction> fusion_op_kind = {};
  };
  std::unordered_map<framework::OpPatternKind, FusionRelation> fusion_relation_map_;
};

void InsertBroadcastTo(Graph* graph) {
  // get nodes and op pattern.
  auto graph_nodes      = std::get<0>(graph->topological_order());
  auto& op_pattern_dict = framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
  auto& dtype_dict      = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict      = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  // node index.
  auto index = graph_nodes.size();
  for (auto graph_node : graph_nodes) {
    auto node = graph_node->safe_as<Node>();
    // if node is NodeData, continue.
    if (!node) {
      continue;
    }
    // check kBroadcast op and insert broadcast to.
    if (op_pattern_dict[node->op()] == framework::kBroadcast && node->op()->name != "broadcast_to") {
      // get output shape
      auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
      CHECK(node_data);
      CHECK(shape_dict.count(node_data->id())) << "Can't find " << node_data->id() << " 's shape!";
      auto output_shape = shape_dict.at(node_data->id());

      // check input node
      for (auto& edge : node->inlinks_in_order()) {
        auto input_data = edge->source()->safe_as<NodeData>();
        CHECK(input_data);
        CHECK(shape_dict.count(input_data->id())) << "Can't find " << input_data->id() << " 's shape!";
        auto input_shape = shape_dict.at(input_data->id());
        // input shape is not equal to output shape, insert broadcast_to
        if (output_shape != input_shape) {
          // input_data UnLinkTo node
          input_data->UnLinkSingleTo(node);
          int axis = -1;
          if (node->attrs.attr_store.find("axis") != node->attrs.attr_store.end()) {
            axis = absl::get<int>(node->attrs.attr_store["axis"]);
          }
          if (axis == -1) {
            axis = output_shape.size() - 1;
          }
          node->attrs.attr_store = {};
          CHECK_LE(axis + input_shape.size(), output_shape.size());
          // create node
          auto tmp_node = new Node(
              framework::Operator::Get("broadcast_to"), "broadcast_to", "broadcast_to_" + std::to_string(++index));
          std::vector<int> broadcast_axes;
          for (int idx = 0; idx < input_shape.size(); ++idx) {
            broadcast_axes.push_back(axis++);
          }
          tmp_node->attrs.attr_store["out_shape"]      = output_shape;
          tmp_node->attrs.attr_store["broadcast_axes"] = broadcast_axes;
          input_data->LinkTo(tmp_node);
          graph->RegisterNode(tmp_node->id(), tmp_node);
          // create node data
          auto tmp_node_data = new NodeData(nullptr, 0, 0, "var_" + std::to_string(index), false);
          tmp_node->LinkTo(tmp_node_data);
          tmp_node_data->LinkTo(node);
          graph->RegisterNode(tmp_node_data->id(), tmp_node_data);
          // update shape_dict
          shape_dict[tmp_node_data->id()] = output_shape;
          // update dtype_dict
          dtype_dict[tmp_node_data->id()] = dtype_dict[node_data->id()];
        }
      }
    }
  }
}

void OpFusionPassInternal(Graph* graph) {
  InsertBroadcastTo(graph);
  // nodes include(node, data node)
  auto nodes = std::get<0>(graph->topological_order());
  // shape
  auto& shape_dict = graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  auto op_fusion_helper = OpFusionPassHelper(nodes, shape_dict);
  graph->fusion_groups  = op_fusion_helper();

  for (auto& Group : graph->fusion_groups) {
    VLOG(11) << "Group Start.";
    for (auto node : Group->nodes) {
      VLOG(11) << "node -> " << node->id();
    }
    VLOG(11) << "Group End.";
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(OpFusionPass) {
  CINN_REGISTER_PASS(OpFusionPass)
      .describe(
          "Op Fusion Pass which performs Ops fusion, Producer Ops are fused into Consumer Ops with certain conditions.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::OpFusionPassInternal);

  return true;
}

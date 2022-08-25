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

#include <absl/container/flat_hash_map.h>

#include <deque>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/visualize_helper.h"
#include "cinn/hlir/pass/check_fusion_accuracy_pass_util.h"

namespace cinn::hlir::pass {

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::NodePtr;
using framework::Operator;

using common::GraphEdge;
using common::GraphNode;

using GroupPtr  = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

using ShapeDict = absl::flat_hash_map<std::string, framework::shape_t>;
using DtypeDict = absl::flat_hash_map<std::string, common::Type>;

class CheckFusionAccuracyPass {
 public:
  CheckFusionAccuracyPass(Graph* graph_)
      : graph_(graph_),
        shape_dict_(graph_->GetMutableAttrs<ShapeDict>("infershape")),
        dtype_dict_(graph_->GetMutableAttrs<DtypeDict>("inferdtype")) {}

  GroupList Apply();

 protected:
  // a helper find to get node data's debug information
  std::string DebugNodeData(NodeData* node);

  // create node's output node, whose name is output_id
  NodeData* CreateOutputNode(NodePtr node, const std::string& output_id = "");

  // create a group, the group only has one node
  GroupPtr CreateSingleNodeGroup(NodePtr node_ptr);

  // topological order nodes list
  std::vector<Node*> TopologicalOrder(const std::vector<Node*>& nodes);

  // copy and create new output from old_node, and link to new_node
  void CreateCheckNodeOutputs(Node* old_node, NodePtr new_node);

  // relink new_node's input node
  void RelinkNodeInputs(Node* old_node, NodePtr new_node);

  // create check fusion accuracy pass node
  NodePtr CreateCheckNode(Node* node);

  std::pair<NodePtr, NodeData*> CreateIsCloseNode(const std::string& node_id);

  std::pair<NodePtr, NodeData*> CreateAllNode(const std::string& node_id);

  std::pair<NodePtr, NodeData*> CreateAssertNode(const std::string& node_id, const std::string& assert_msg);

  // the AssertAllClose operator are composed of isclose+all+assert
  std::vector<NodePtr> CreateAssertAllClose(const std::string& node_id,
                                            const std::string& assert_msg,
                                            const std::vector<NodeData*>& inputs);

  // link origin group's output and pass group's output to the AssertAllClose nodes
  GroupList LinkToAssertAllClose(const std::unordered_set<NodeData*>& group_outputs, utils::AssertMsg* msg);

 private:
  Graph* graph_;
  std::unordered_map<NodeData*, NodeData*> old2new_nodedata_map_;

  ShapeDict& shape_dict_;
  DtypeDict& dtype_dict_;
};

std::string CheckFusionAccuracyPass::DebugNodeData(NodeData* node) {
  std::stringstream ss;
  ss << node->id() << "{shape=[" << cinn::utils::Join(shape_dict_.at(node->id()), ", ")
     << "], dtype=" << dtype_dict_.at(node->id()) << "}";
  return ss.str();
}

NodeData* CheckFusionAccuracyPass::CreateOutputNode(NodePtr node, const std::string& output_id) {
  // create node's output data node
  auto node_id = output_id;
  if (node_id.empty()) {
    node_id = "var_" + node->id();
  }

  auto graph_node = graph_->RetrieveNode(node_id);
  CHECK(graph_node == nullptr) << "The node " << node->op()->name << "'s output" << node_id
                               << " had been registered in graph! Please check.";

  auto* output_data = new NodeData(node, 0, 0, node_id);
  node->LinkTo(output_data);
  graph_->RegisterNode(node_id, output_data);

  return output_data;
}

void CheckFusionAccuracyPass::CreateCheckNodeOutputs(Node* old_node, NodePtr new_node) {
  const auto& outlinks = old_node->outlinks_in_order();
  for (const auto& out_edge : outlinks) {
    auto out_node = out_edge->sink()->safe_as<NodeData>();
    CHECK(out_node) << "Node " << old_node->id() << "'s output node is nullptr! Please check.";

    const auto& out_node_id = out_node->id();
    // If the check node's output variable node not created
    if (!old2new_nodedata_map_.count(out_node)) {
      const auto& check_out_node_id = utils::GenerateCheckFusionAccuracyNodeId(out_node_id);

      auto check_out_node          = CreateOutputNode(new_node, check_out_node_id);
      check_out_node->output_index = out_node->output_index;

      // why assign twice? only assign once may cause useless assign, a puzzling bug
      shape_dict_[check_out_node_id] = shape_dict_.at(out_node_id);
      shape_dict_[check_out_node_id] = shape_dict_.at(out_node_id);

      dtype_dict_[check_out_node_id] = dtype_dict_.at(out_node_id);
      dtype_dict_[check_out_node_id] = dtype_dict_.at(out_node_id);

      VLOG(4) << "Create the check fusion accuracy node of node " << old_node->id() << "'s output node "
              << DebugNodeData(out_node) << " success, which is " << DebugNodeData(check_out_node);

      old2new_nodedata_map_[out_node] = check_out_node;
    }
  }
}

void CheckFusionAccuracyPass::RelinkNodeInputs(Node* old_node, NodePtr new_node) {
  const auto& inlinks = old_node->inlinks_in_order();
  for (const auto& in_edge : inlinks) {
    auto in_node = in_edge->source()->safe_as<NodeData>();
    CHECK(in_node) << "Node " << old_node->id() << "'s input node is nullptr! Please check.";

    if (old2new_nodedata_map_.count(in_node)) {
      old2new_nodedata_map_[in_node]->LinkTo(new_node.get());
    } else {
      in_node->LinkTo(new_node.get());
    }
  }
}

NodePtr CheckFusionAccuracyPass::CreateCheckNode(Node* node) {
  CHECK(node->op()) << "Node " << node->id() << " is not operator! Please check.";

  const auto& check_node_id = utils::GenerateCheckFusionAccuracyNodeId(node->id());

  auto check_node              = Node::Create(node->op(), node->attrs.node_name, check_node_id);
  check_node->attrs.attr_store = node->attrs.attr_store;

  graph_->RegisterNode(check_node_id, check_node.get());

  CreateCheckNodeOutputs(node, check_node);
  RelinkNodeInputs(node, check_node);

  VLOG(4) << "Create node " << framework::DebugString(node) << "'s check fusion accuracy node success, which is "
          << framework::DebugString(check_node.get());

  return check_node;
}

GroupPtr CheckFusionAccuracyPass::CreateSingleNodeGroup(NodePtr node_ptr) {
  auto node = node_ptr.get();

  // init group
  auto group = std::make_shared<Graph::Group>();

  group->nodes.push_back(node);
  group->nodes_set.insert(node);

  // output nodes
  for (auto& edge : node->outlinks_in_order()) {
    auto output_node_data = edge->sink()->safe_as<NodeData>();
    CHECK(output_node_data) << "Node " << node->id() << "'s output node is nullptr! Please check.";
    for (auto& data_edge : output_node_data->outlinks()) {
      group->output_nodes.insert(data_edge->sink()->safe_as<Node>());
    }
  }

  // input nodes
  for (auto& edge : node->inlinks_in_order()) {
    auto input_node_data = edge->source()->safe_as<NodeData>();
    CHECK(input_node_data) << "Node " << node->id() << "'s input node is nullptr! Please check.";

    // input data has source node
    if (input_node_data->source_node.get()) {
      group->input_nodes[input_node_data->source_node.get()] = 1;
    }
  }

  // group type
  group->op_pattern_kind = framework::kOpaque;

  // use current node as master node for schedule
  group->master_nodes.insert(node);
  group->internal_nodes.insert(node);
  group->group_id = node->id();

  return group;
}

std::pair<NodePtr, NodeData*> CheckFusionAccuracyPass::CreateIsCloseNode(const std::string& node_id) {
  const auto& is_close_node_id = "isclose_" + node_id;

  auto is_close_node                      = Node::Create(Operator::Get("isclose"), is_close_node_id, is_close_node_id);
  is_close_node->attrs.attr_store["rtol"] = 1e-05f;
  is_close_node->attrs.attr_store["atol"] = 1e-08f;
  is_close_node->attrs.attr_store["equal_nan"] = false;

  graph_->RegisterNode(is_close_node_id, is_close_node.get());

  // create node's output data node
  auto output_data = CreateOutputNode(is_close_node);

  shape_dict_[output_data->id()] = shape_dict_.at(node_id);
  shape_dict_[output_data->id()] = shape_dict_.at(node_id);

  dtype_dict_[output_data->id()] = common::Bool();
  dtype_dict_[output_data->id()] = common::Bool();

  VLOG(4) << "Create node " << node_id << "'s isclose node success, whose id is " << is_close_node_id
          << ", whose output is " << DebugNodeData(output_data);

  return {is_close_node, output_data};
}

std::pair<NodePtr, NodeData*> CheckFusionAccuracyPass::CreateAllNode(const std::string& node_id) {
  const auto& all_node_id = "all_" + node_id;

  auto all_node = Node::Create(Operator::Get("reduce_all"), all_node_id, all_node_id);

  int shape_size = shape_dict_[node_id].size();
  std::vector<int> axes(shape_size);
  for (int i = 0; i < shape_size; ++i) {
    axes[i] = i;
  }
  all_node->attrs.attr_store["dim"]      = axes;
  all_node->attrs.attr_store["keep_dim"] = false;

  graph_->RegisterNode(all_node_id, all_node.get());

  // create node's output data node
  auto output_data = CreateOutputNode(all_node);

  shape_dict_[output_data->id()] = framework::shape_t{1};
  shape_dict_[output_data->id()] = framework::shape_t{1};

  dtype_dict_[output_data->id()] = common::Bool();
  dtype_dict_[output_data->id()] = common::Bool();

  VLOG(4) << "Create node " << node_id << "'s all node success, whose id is " << all_node_id << ", whose output is "
          << DebugNodeData(output_data);

  return {all_node, output_data};
}

std::pair<NodePtr, NodeData*> CheckFusionAccuracyPass::CreateAssertNode(const std::string& node_id,
                                                                        const std::string& assert_msg) {
  const auto& assert_node_id = "assert_" + node_id;

  auto assert_node                     = Node::Create(Operator::Get("assert_true"), assert_node_id, assert_node_id);
  assert_node->attrs.attr_store["msg"] = assert_msg;
  assert_node->attrs.attr_store["only_warning"] = false;

  graph_->RegisterNode(assert_node_id, assert_node.get());

  // create node's output data node
  auto output_data = CreateOutputNode(assert_node);

  shape_dict_[output_data->id()] = framework::shape_t{1};
  shape_dict_[output_data->id()] = framework::shape_t{1};

  dtype_dict_[output_data->id()] = common::Bool();
  dtype_dict_[output_data->id()] = common::Bool();

  VLOG(4) << "Create node " << node_id << "'s assert node success, whose id is " << assert_node_id
          << ", whose output is " << DebugNodeData(output_data);

  return {assert_node, output_data};
}

std::vector<NodePtr> CheckFusionAccuracyPass::CreateAssertAllClose(const std::string& node_id,
                                                                   const std::string& assert_msg,
                                                                   const std::vector<NodeData*>& inputs) {
  // create isclose + all + assert nodes
  const auto& is_close_node = CreateIsCloseNode(node_id);
  const auto& all_node      = CreateAllNode(node_id);
  const auto& assert_node   = CreateAssertNode(node_id, assert_msg);

  // link each nodes, each node only has one output
  for (auto in_data : inputs) {
    in_data->LinkTo(is_close_node.first.get());
  }
  is_close_node.second->LinkTo(all_node.first.get());
  all_node.second->LinkTo(assert_node.first.get());

  return {is_close_node.first, all_node.first, assert_node.first};
}

GroupList CheckFusionAccuracyPass::LinkToAssertAllClose(const std::unordered_set<NodeData*>& group_outputs,
                                                        utils::AssertMsg* msg) {
  GroupList assert_groups;
  for (auto group_out : group_outputs) {
    CHECK(old2new_nodedata_map_.count(group_out)) << "The check fusion accuracy's node corresponding to "
                                                  << group_out->id() << " had not been created! Please check.";
    auto pass_out = old2new_nodedata_map_.at(group_out);

    msg->SetMsg("Var id", group_out->id());

    const auto& nodes = CreateAssertAllClose(pass_out->id(), msg->str(), {group_out, pass_out});

    for (const auto& node : nodes) {
      assert_groups.emplace_back(CreateSingleNodeGroup(node));
    }
  }
  return assert_groups;
}

std::vector<Node*> CheckFusionAccuracyPass::TopologicalOrder(const std::vector<Node*>& nodes) {
  std::vector<Node*> ordered_nodes;

  // count all node's output to find the group's start node
  std::unordered_set<NodeData*> all_outputs;
  for (auto node : nodes) {
    for (auto& out_edge : node->outlinks_in_order()) {
      all_outputs.insert(out_edge->sink()->safe_as<NodeData>());
    }
  }

  // if the node's input is not any group node's output, it's start node
  std::deque<Node*> queue;
  std::unordered_map<Node*, int> indegree;
  for (auto node : nodes) {
    bool is_start = true;
    for (auto& in_edge : node->inlinks_in_order()) {
      if (all_outputs.count(in_edge->source()->safe_as<NodeData>())) {
        // if the node's input is some group node's output, it's not start node
        is_start = false;
        indegree[node]++;
      }
    }
    if (is_start) {
      queue.emplace_back(node);
    }
  }

  // start to visit
  while (!queue.empty()) {
    auto top_node = queue.front();
    ordered_nodes.push_back(top_node);

    queue.pop_front();

    for (auto& out_edge : top_node->outlinks_in_order()) {
      // the output of node is a variable node, not op node
      auto out_data = out_edge->sink()->safe_as<NodeData>();

      for (auto out_data_edge : out_data->outlinks()) {
        // the variable node's output are the required output nodes
        auto out_node = out_data_edge->sink()->safe_as<Node>();
        if (indegree.count(out_node) && (--indegree[out_node]) == 0) {
          // if the output node in group and its input nodes are all visited, push
          queue.push_back(out_node);
        }
      }
    }
  }

  CHECK_EQ(ordered_nodes.size(), nodes.size()) << "There has circle in group! Please check.";

  return ordered_nodes;
}

GroupList CheckFusionAccuracyPass::Apply() {
  GroupList check_fusion_groups;

  auto serial_name = [&](const std::unordered_set<NodeData*>& nodes) {
    std::string res;
    std::for_each(nodes.begin(), nodes.end(), [&](NodeData* node) { res += DebugNodeData(node) + ", "; });
    return res;
  };

  for (auto& group : graph_->fusion_groups) {
    check_fusion_groups.emplace_back(group);

    const auto& group_nodes = group->CollectNodes();

    // fusion group only has one node, do not need check, skip
    if (group_nodes.size() <= 1) {
      VLOG(4) << "The Group " << group->GetFuncName() << " just has one node, skip.";
      continue;
    }

    // split orign group and create group for each node
    const auto& ordered_nodes = TopologicalOrder(group_nodes);
    VLOG(4) << "Check the accuracy of group " << graph_->DebugGroupedGraph({ordered_nodes});

    for (auto* node : ordered_nodes) {
      if (node->is_variable()) {
        VLOG(4) << "The node " << node->id() << " is variable, skip check fusion accuracy.";
        continue;
      }

      auto check_node = CreateCheckNode(node);
      check_fusion_groups.push_back(CreateSingleNodeGroup(check_node));
    }

    // get group's output data node list
    auto input_datas  = group->GetInputNodeDatas();
    auto output_datas = group->GetOutputNodeDatas(graph_->outputs);

    // set assert debug info
    utils::AssertMsg msg("check accuracy of kernel " + group->GetFuncName());
    msg.SetMsg("Kernel name", group->GetFuncName());
    msg.SetMsg("Input list", serial_name(input_datas));
    msg.SetMsg("Output list", serial_name(output_datas));
    msg.SetMsg("Group graph", graph_->DebugGroupedGraph({ordered_nodes}));

    // link the group's output data node to assert all close node
    const auto& assert_group = LinkToAssertAllClose(output_datas, &msg);
    check_fusion_groups.insert(check_fusion_groups.end(), assert_group.begin(), assert_group.end());
  }
  return check_fusion_groups;
}

void CheckFusionAccuracyPassImpl(Graph* graph) {
  VLOG(3) << "Before CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph();

  graph->fusion_groups = CheckFusionAccuracyPass(graph).Apply();

  VLOG(3) << "After CheckFusionAccuracyPass:\n" << graph->DebugGroupedGraph();
}

}  // namespace cinn::hlir::pass

CINN_REGISTER_HELPER(CheckFusionAccuracyPass) {
  CINN_REGISTER_PASS(CheckFusionAccuracyPass)
      .describe("Check Fusion Accuracy Pass.")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::CheckFusionAccuracyPassImpl);

  return true;
}

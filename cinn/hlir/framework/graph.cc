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

#include "cinn/hlir/framework/graph.h"

#include <atomic>
#include <sstream>

#include "cinn/hlir/framework/visualize_helper.h"
#include "cinn/utils/string.h"

DECLARE_string(cinn_fusion_groups_graphviz_dir);

namespace cinn {
namespace hlir {
namespace framework {

void Graph::Initialize(const frontend::Program& prog,
                       const std::unordered_set<std::string>& fetch_var_ids,
                       const Target& target) {
  target_ = target;
  absl::flat_hash_map<std::string, shape_t> shape_dict;
  absl::flat_hash_map<std::string, common::Type> dtype_dict;
  int counter = 0;
  for (size_t i = 0; i < prog.size(); i++) {
    auto temp = prog[i];
    Node* node_tmp =
        new Node(Operator::Get(temp->op_type), temp->op_type, temp->op_type + "_" + std::to_string(counter++));
    Shared<Node> node_ptr(node_tmp);
    node_tmp->attrs.attr_store = temp->attrs;
    for (auto& input_v : temp->inputs) {
      common::GraphNode* graph_node = this->RetrieveNode(input_v->id);
      if (!graph_node) {
        dtype_dict[input_v->id] = input_v->type;
        shape_dict[input_v->id] = input_v->shape;
        NodeData* input_data    = new NodeData(nullptr, 0, 0, input_v->id, input_v.is_const());
        input_data->LinkTo(node_tmp);
        this->RegisterNode(input_v->id, input_data);
      } else {
        graph_node->as<NodeData>()->LinkTo(node_tmp);
      }
    }
    int out_idx = 0;
    for (auto& output_v : temp->outputs) {
      common::GraphNode* graph_node = this->RetrieveNode(output_v->id);
      if (!graph_node) {
        dtype_dict[output_v->id] = output_v->type;
        shape_dict[output_v->id] = output_v->shape;
        auto* output_data        = new NodeData(node_ptr, out_idx++, 0, output_v->id);
        if (fetch_var_ids.count(output_v->id)) {
          outputs.push_back(output_data);
        }
        node_tmp->LinkTo(output_data);
        this->RegisterNode(output_v->id, output_data);
      } else {
        node_tmp->LinkTo(graph_node->as<NodeData>());
        graph_node->as<NodeData>()->set_const(false);
        graph_node->as<NodeData>()->output_index = out_idx++;
        graph_node->as<NodeData>()->source_node  = node_ptr;
      }
    }
    this->RegisterNode(node_tmp->id(), node_tmp);
  }
  this->attrs["infershape"] = std::make_shared<absl::any>(shape_dict);
  this->attrs["inferdtype"] = std::make_shared<absl::any>(dtype_dict);
}

std::vector<std::vector<Node*>> Graph::FusionGroupsToGroups() {
  std::vector<std::vector<Node*>> groups;
  groups.resize(fusion_groups.size());
  for (size_t i = 0; i < fusion_groups.size(); ++i) {
    groups[i] = fusion_groups[i]->CollectNodes();
  }
  return groups;
}

std::string Graph::DebugGroupedGraph(const std::unordered_set<std::string>& fetch_var_ids) {
  return DebugGroupedGraph(FusionGroupsToGroups(), fetch_var_ids);
}

std::string Graph::DebugGroupedGraph(const std::vector<std::vector<Node*>>& groups,
                                     const std::unordered_set<std::string>& fetch_var_ids) {
  std::stringstream debug_str;
  for (auto& id : fetch_var_ids) {
    debug_str << "Fetch: " << id << "\n";
  }

  int group_id = 0;
  for (auto& group : groups) {
    debug_str << "Group " << group_id++ << " {\n";
    for (auto* node : group) {
      debug_str << "  " << DebugString(node) << "\n";
    }
    debug_str << "}"
              << "\n";
  }
  return debug_str.str();
}

void Graph::VisualizeGroupedGraph(const std::unordered_set<std::string>& fetch_var_ids) {
  VisualizeGroupedGraph(FusionGroupsToGroups(), fetch_var_ids);
}

void Graph::VisualizeGroupedGraph(const std::vector<std::vector<Node*>>& groups,
                                  const std::unordered_set<std::string>& fetch_var_ids) {
  VLOG(4) << DebugGroupedGraph(groups, fetch_var_ids);

  if (FLAGS_cinn_fusion_groups_graphviz_dir.empty()) {
    return;
  }

  viz_path_ = utils::StringFormat(
      "%s/fusion_groups_%d/", FLAGS_cinn_fusion_groups_graphviz_dir.c_str(), viz_count_.fetch_add(1));
  VLOG(4) << "The visualized path of CINN fusion groups: " << viz_path_;
  if (!MakeDirectory(viz_path_, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
    return;
  }

  Summary(groups, viz_path_);

  auto& shape_dict = HasAttr("infershape") ? GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape")
                                           : absl::flat_hash_map<std::string, shape_t>{};

  std::unordered_map<std::string, int> recompute_nodes;
  FindRecomputeNodes(groups, &recompute_nodes);

  utils::DotLang dot;
  utils::ResetDotCounters();

  // Record the NodeData's actually ids.
  std::unordered_set<std::string> nodedatas_set;

  int group_id = 0;
  for (auto& group : groups) {
    std::string dot_cluster_id = GenClusterId(group, group_id);
    dot.AddCluster(dot_cluster_id, GetGroupAttrs(group.size()));

    std::unordered_map<std::string, std::string> outnode2dot_id;
    for (auto* node : group) {
      AddGroupNode(
          node, dot_cluster_id, fetch_var_ids, shape_dict, &recompute_nodes, &outnode2dot_id, &nodedatas_set, &dot);
    }
    group_id++;
  }

  std::string filepath = viz_path_ + "grouped_graph.dot";
  WriteToFile(filepath, dot());

  VisualizeGroups(groups, fetch_var_ids);
}

void Graph::VisualizeGroups(const std::vector<std::vector<Node*>>& groups,
                            const std::unordered_set<std::string>& fetch_var_ids) {
  auto& shape_dict = HasAttr("infershape") ? GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape")
                                           : absl::flat_hash_map<std::string, shape_t>{};

  std::unordered_map<std::string, int> recompute_nodes;
  FindRecomputeNodes(groups, &recompute_nodes);

  utils::ResetDotCounters();

  int group_id = 0;
  for (auto& group : groups) {
    utils::DotLang dot;
    std::unordered_set<Node*> nodes_set;
    std::string dot_cluster_id = GenClusterId(group, group_id);
    dot.AddCluster(dot_cluster_id, GetGroupAttrs(group.size()));

    std::unordered_map<std::string, std::string> outnode2dot_id;
    for (auto* node : group) {
      AddGroupNode(node, dot_cluster_id, fetch_var_ids, shape_dict, &recompute_nodes, &outnode2dot_id, nullptr, &dot);
      nodes_set.insert(node);
    }

    for (auto& node : group) {
      for (auto& inlink : node->inlinks()) {
        auto* innode = inlink->source()->safe_as<NodeData>();
        if (innode) {
          std::string dot_innode_id = outnode2dot_id[innode->id()];
          for (auto& innode_inlink : innode->inlinks()) {
            auto* in_innode = innode_inlink->source()->safe_as<Node>();
            if (in_innode && !nodes_set.count(in_innode)) {
              nodes_set.insert(in_innode);
              dot.AddNode(in_innode->id(), GetOutlinkOpAttrs());
              dot.AddEdge(in_innode->id(), dot_innode_id, {});
            }
          }
        }
      }
      for (auto& outlink : node->outlinks()) {
        auto* outnode = outlink->sink()->safe_as<NodeData>();
        if (outnode) {
          std::string dot_outnode_id = outnode2dot_id[outnode->id()];
          for (auto& outnode_outlink : outnode->outlinks()) {
            auto* out_outnode = outnode_outlink->sink()->safe_as<Node>();
            if (out_outnode && !nodes_set.count(out_outnode)) {
              nodes_set.insert(out_outnode);
              dot.AddNode(out_outnode->id(), GetOutlinkOpAttrs());
              dot.AddEdge(dot_outnode_id, out_outnode->id(), {});
            }
          }
        }
      }
    }

    std::string filepath = GetFilePathForGroup(groups, group_id, viz_path_);
    WriteToFile(filepath, dot());

    group_id++;
  }
}

std::atomic_size_t Graph::viz_count_{0};

std::unordered_set<NodeData*> Graph::Group::GetInputNodeDatas() {
  std::unordered_set<NodeData*> group_inputs;

  // count all node's input data
  for (auto node : this->CollectNodes()) {
    for (auto& in_edge : node->inlinks_in_order()) {
      auto input_data = in_edge->source()->safe_as<NodeData>();
      if (!input_data) {
        continue;
      }

      if (!input_data->source_node.get()) {
        // if the input data hasn't input op, it's the group's input
        group_inputs.insert(input_data);
        continue;
      }

      if (std::find(this->input_names.begin(), this->input_names.end(), input_data->id()) != this->input_names.end()) {
        // if the input data in group's input_names
        group_inputs.insert(input_data);
        continue;
      }
    }
  }

  return group_inputs;
}

std::unordered_set<NodeData*> Graph::Group::GetOutputNodeDatas(const std::vector<NodeData*>& graph_outputs) {
  std::unordered_set<NodeData*> group_outputs;

  auto group_nodes = this->CollectNodes();
  auto nodes       = std::unordered_set<Node*>(group_nodes.begin(), group_nodes.end());
  // count all node's output data
  for (auto node : group_nodes) {
    for (auto& out_edge : node->outlinks_in_order()) {
      auto output_data = out_edge->sink()->safe_as<NodeData>();
      if (!output_data) {
        continue;
      }

      if (std::find(graph_outputs.begin(), graph_outputs.end(), output_data) != graph_outputs.end()) {
        // if the output data is the graph's output data, it's also the group's output
        group_outputs.insert(output_data);
        continue;
      }

      if (std::find(this->output_names.begin(), this->output_names.end(), output_data->id()) !=
          this->output_names.end()) {
        // if the output data in group's output_names
        group_outputs.insert(output_data);
        continue;
      }

      for (auto out_edge : output_data->outlinks()) {
        auto out_node = out_edge->sink()->safe_as<Node>();

        // check whether the output data's output op in group.output_nodes,
        // if true, the output data is the group's output data
        if (!nodes.count(out_node)) {
          group_outputs.insert(output_data);
        }
      }
    }
  }

  for (auto node : this->output_nodes) {
    for (auto& link : node->outlinks_in_order(true)) {
      auto node_data = link->sink()->safe_as<NodeData>();
      if (!node_data) {
        continue;
      }

      group_outputs.insert(node_data);
    }
  }

  return group_outputs;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

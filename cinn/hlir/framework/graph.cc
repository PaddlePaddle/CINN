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
    VLOG(3) << "operator [" << temp->op_type << "] has [" << temp->inputs.size() << "] inputs, and ["
            << temp->outputs.size() << "] outputs";
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

void Graph::VisualizeGroupedGraph(const std::unordered_set<std::string>& fetch_var_ids) {
  std::vector<std::vector<Node*>> groups;
  groups.resize(fusion_groups.size());
  for (size_t i = 0; i < fusion_groups.size(); ++i) {
    groups[i] = fusion_groups[i]->CollectNodes();
  }
  VisualizeGroupedGraph(groups, fetch_var_ids);
}

void Graph::VisualizeGroupedGraph(const std::vector<std::vector<Node*>>& groups,
                                  const std::unordered_set<std::string>& fetch_var_ids) {
  if (FLAGS_cinn_fusion_groups_graphviz_dir.empty()) {
    return;
  }

  viz_path_ = utils::StringFormat(
      "%s/fusion_groups_%d/", FLAGS_cinn_fusion_groups_graphviz_dir.c_str(), viz_count_.fetch_add(1));
  VLOG(4) << "The visualized path of CINN fusion groups: " << viz_path_;
  if (!MakeDirectory(viz_path_, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
    return;
  }

  for (auto& id : fetch_var_ids) {
    VLOG(4) << "Fetch: " << id;
  }

  int group_id = 0;
  for (auto& group : groups) {
    VLOG(4) << "Group " << group_id++ << " {";
    for (auto* node : group) {
      VLOG(4) << "  " << DebugString(node);
    }
    VLOG(4) << "}";
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

  group_id = 0;
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

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

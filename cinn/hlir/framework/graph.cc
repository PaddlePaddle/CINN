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

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "cinn/utils/dot_lang.h"

DECLARE_string(cinn_fusion_groups_graphviz_dir);

namespace cinn {
namespace hlir {
namespace framework {

Graph::Graph(const frontend::Program& prog, const Target& target) {
  target_ = target;
  absl::flat_hash_map<std::string, shape_t> shape_dict;
  absl::flat_hash_map<std::string, common::Type> dtype_dict;
  int counter = 0;
  for (size_t i = 0; i < prog.size(); i++) {
    auto temp = prog[i];
    Node* node_tmp =
        new Node(Operator::Get(temp->op_type), temp->op_type, temp->op_type + "_" + std::to_string(counter++));
    std::shared_ptr<Node> node_ptr(node_tmp);
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
      dtype_dict[output_v->id] = output_v->type;
      shape_dict[output_v->id] = output_v->shape;
      auto* output_data        = new NodeData(node_ptr, out_idx++, 0, output_v->id);
      node_tmp->LinkTo(output_data);
      this->RegisterNode(output_v->id, output_data);
    }
    this->RegisterNode(node_tmp->id(), node_tmp);
  }
  this->attrs["infershape"] = std::make_shared<absl::any>(shape_dict);
  this->attrs["inferdtype"] = std::make_shared<absl::any>(dtype_dict);
}

bool MakeDirectory(const std::string& dirname) {
  if (access(dirname.c_str(), F_OK)) {
    if (mkdir(dirname.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO) != 0) {
      return false;
    }
  }
  return true;
}

std::string GetFilePath(const std::vector<Node*>& group, int group_id) {
  std::string filename = std::to_string(group_id);
  for (auto* node : group) {
    filename += "_" + node->id();
  }

  int max_len                     = 50;
  std::string simplified_filename = filename;
  if (filename.size() > max_len) {
    std::unordered_map<std::string, std::string> funcname_map = {{"const_scalar", "scalar"},
                                                                 {"fill_constant", "fill"},
                                                                 {"identity", "copy"},
                                                                 {"broadcast_to", "broadcast"},
                                                                 {"elementwise_add", "add"},
                                                                 {"substract", "sub"},
                                                                 {"elementwise_mul", "mul"},
                                                                 {"divide", "div"},
                                                                 {"reduce_sum", "reduce"}};
    for (auto& item : funcname_map) {
      size_t index = 0;
      while (true) {
        index = simplified_filename.find(item.first, index);
        if (index == std::string::npos) {
          break;
        }
        simplified_filename.replace(index, item.first.size(), item.second);
        index += item.second.size();
      }
    }
  }

  return FLAGS_cinn_fusion_groups_graphviz_dir + "/" + simplified_filename.substr(0, 50) + ".dot";
}

std::string GenClusterId(const std::vector<Node*>& group, int group_id) {
  return "group_" + std::to_string(group_id) + "(" + std::to_string(group.size()) + ")";
}

std::vector<utils::Attr> GetGroupOpAttrs() {
  return std::vector<utils::Attr>{
      utils::Attr("shape", "Mrecord"), utils::Attr("color", "#8EABFF"), utils::Attr("style", "filled")};
}

std::vector<utils::Attr> GetOutlinkOpAttrs() {
  return std::vector<utils::Attr>{
      utils::Attr("shape", "Mrecord"), utils::Attr("color", "#ff7f00"), utils::Attr("style", "filled")};
}

std::vector<utils::Attr> GetGroupVarAttrs() {
  return std::vector<utils::Attr>{utils::Attr("color", "#FFDC85"), utils::Attr("style", "filled")};
}

std::vector<utils::Attr> GetFetchVarAttrs() {
  return std::vector<utils::Attr>{
      utils::Attr("peripheries", "2"), utils::Attr("color", "#43CD80"), utils::Attr("style", "filled")};
}

std::vector<utils::Attr> GetGroupAttrs(size_t group_size) {
  std::string fillcolor;
  if (group_size == 1) {
    fillcolor = "#E8E8E8";
  } else if (group_size <= 3) {
    fillcolor = "#FFFFF0";
  } else if (group_size <= 10) {
    fillcolor = "#F0FFFF";
  } else {
    // group_size > 10
    fillcolor = "#EEE5DE";
  }
  std::vector<utils::Attr> attrs = {
      utils::Attr("color", "grey"), utils::Attr("style", "filled"), utils::Attr("fillcolor", fillcolor)};
  return attrs;
}

void Graph::VisualizeGroupedGraph(const std::vector<std::vector<Node*>>& groups,
                                  const std::unordered_set<std::string>& fetch_var_ids) {
  if (FLAGS_cinn_fusion_groups_graphviz_dir.empty()) {
    return;
  }

  if (!MakeDirectory(FLAGS_cinn_fusion_groups_graphviz_dir)) {
    return;
  }

  std::vector<utils::Attr> group_op_attrs  = GetGroupOpAttrs();
  std::vector<utils::Attr> group_var_attrs = GetGroupVarAttrs();
  std::vector<utils::Attr> fetch_var_attrs = GetFetchVarAttrs();

  utils::DotLang dot;
  std::unordered_set<NodeData*> nodedatas_set;

  int group_id = 0;
  for (auto& group : groups) {
    std::string cluster_id = GenClusterId(group, group_id);
    dot.AddCluster(cluster_id, GetGroupAttrs(group.size()));
    for (auto& node : group) {
      dot.AddNode(node->id(), group_op_attrs, "", cluster_id);
      for (auto& inlink : node->inlinks()) {
        auto* innode = inlink->source()->safe_as<NodeData>();
        if (innode) {
          if (!nodedatas_set.count(innode)) {
            dot.AddNode(innode->id(), group_var_attrs, "", cluster_id, true);
            nodedatas_set.insert(innode);
          }
          dot.AddEdge(innode->id(), node->id(), {});
        }
      }
      for (auto& outlink : node->outlinks()) {
        auto* outnode = outlink->sink()->safe_as<NodeData>();
        if (outnode) {
          if (!nodedatas_set.count(outnode)) {
            if (fetch_var_ids.count(outnode->id())) {
              dot.AddNode(outnode->id(), fetch_var_attrs, "", cluster_id, true);
            } else {
              dot.AddNode(outnode->id(), group_var_attrs, "", cluster_id, true);
            }
            nodedatas_set.insert(outnode);
          }
          dot.AddEdge(node->id(), outnode->id(), {});
        }
      }
    }
    group_id++;
  }

  std::string filepath = FLAGS_cinn_fusion_groups_graphviz_dir + "/grouped_graph.dot";
  VLOG(4) << "Write to " << filepath;
  std::ofstream of(filepath);
  of << dot();
  of.close();

  VisualizeGroups(groups, fetch_var_ids);
}

void Graph::VisualizeGroups(const std::vector<std::vector<Node*>>& groups,
                            const std::unordered_set<std::string>& fetch_var_ids) {
  std::vector<utils::Attr> group_op_attrs  = GetGroupOpAttrs();
  std::vector<utils::Attr> group_var_attrs = GetGroupVarAttrs();
  std::vector<utils::Attr> fetch_var_attrs = GetFetchVarAttrs();
  std::vector<utils::Attr> out_op_attrs    = GetOutlinkOpAttrs();

  int group_id = 0;
  for (auto& group : groups) {
    utils::DotLang dot;
    std::unordered_set<Node*> nodes_set;
    std::string cluster_id = GenClusterId(group, group_id);
    dot.AddCluster(cluster_id, GetGroupAttrs(group.size()));
    for (auto& node : group) {
      nodes_set.insert(node);
      dot.AddNode(node->id(), group_op_attrs, "", cluster_id);
      for (auto& inlink : node->inlinks()) {
        auto* innode = inlink->source()->safe_as<NodeData>();
        if (innode) {
          dot.AddNode(innode->id(), group_var_attrs, "", cluster_id, true);
          dot.AddEdge(innode->id(), node->id(), {});
        }
      }
      for (auto& outlink : node->outlinks()) {
        auto* outnode = outlink->sink()->safe_as<NodeData>();
        if (outnode) {
          if (fetch_var_ids.count(outnode->id())) {
            dot.AddNode(outnode->id(), fetch_var_attrs, "", cluster_id, true);
          } else {
            dot.AddNode(outnode->id(), group_var_attrs, "", cluster_id, true);
          }
          dot.AddEdge(node->id(), outnode->id(), {});
        }
      }
    }
    for (auto& node : group) {
      for (auto& inlink : node->inlinks()) {
        auto* innode = inlink->source()->safe_as<NodeData>();
        if (innode) {
          for (auto& innode_inlink : innode->inlinks()) {
            auto* in_innode = innode_inlink->source()->safe_as<Node>();
            if (in_innode && !nodes_set.count(in_innode)) {
              nodes_set.insert(in_innode);
              dot.AddNode(in_innode->id(), out_op_attrs);
              dot.AddEdge(in_innode->id(), innode->id(), {});
            }
          }
        }
      }
      for (auto& outlink : node->outlinks()) {
        auto* outnode = outlink->sink()->safe_as<NodeData>();
        if (outnode) {
          for (auto& outnode_outlink : outnode->outlinks()) {
            auto* out_outnode = outnode_outlink->sink()->safe_as<Node>();
            if (out_outnode && !nodes_set.count(out_outnode)) {
              nodes_set.insert(out_outnode);
              dot.AddNode(out_outnode->id(), out_op_attrs);
              dot.AddEdge(outnode->id(), out_outnode->id(), {});
            }
          }
        }
      }
    }

    std::string filepath = GetFilePath(group, group_id);
    VLOG(4) << "Write to " << filepath;
    std::ofstream of(filepath);
    of << dot();
    of.close();

    group_id++;
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

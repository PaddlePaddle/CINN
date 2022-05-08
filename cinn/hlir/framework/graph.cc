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

#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "cinn/utils/dot_lang.h"
#include "cinn/utils/string.h"

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

bool MakeDirectory(const std::string& dirname, mode_t mode) {
  auto len = dirname.length();
  std::vector<char> dir_path(len + 1, '\0');
  strncpy(dir_path.data(), dirname.c_str(), len);
  char* path = dir_path.data();
  for (char* p = strchr(path + 1, '/'); p; p = strchr(p + 1, '/')) {
    *p = '\0';
    if (mkdir(path, mode) == -1) {
      if (errno != EEXIST) {
        *p = '/';
        return false;
      }
    }
    *p = '/';
  }
  return true;
}

std::string GetFilePathForGroup(const std::vector<std::vector<Node*>>& groups,
                                int group_id,
                                const std::string& viz_path) {
  std::string filename = "";
  for (auto* node : groups[group_id]) {
    filename += "_" + node->id();
  }

  int max_len                     = 50;
  std::string simplified_filename = filename;
  if (filename.size() > max_len) {
    static std::unordered_map<std::string, std::string> funcname_map = {{"const_scalar", "scalar"},
                                                                        {"fill_constant", "fill"},
                                                                        {"identity", "copy"},
                                                                        {"broadcast_to", "broadcast"},
                                                                        {"elementwise_add", "add"},
                                                                        {"substract", "sub"},
                                                                        {"elementwise_mul", "mul"},
                                                                        {"divide", "div"},
                                                                        {"reduce_sum", "reduce"},
                                                                        {"reduce_prod", "reduce"},
                                                                        {"reduce_max", "reduce"},
                                                                        {"reduce_min", "reduce"}};
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

  int width = std::to_string(groups.size()).size();
  std::stringstream ss;
  ss << viz_path;
  ss << std::setw(width) << std::setfill('0') << group_id;
  ss << simplified_filename.substr(0, 50) << ".dot";
  return ss.str();
}

void WriteToFile(const std::string& filepath, const std::string& content) {
  VLOG(4) << "Write to " << filepath;
  std::ofstream of(filepath);
  CHECK(of.is_open()) << "Failed to open " << filepath;
  of << content;
  of.close();
}

std::string GenClusterId(const std::vector<Node*>& group, int group_id) {
  return "group_" + std::to_string(group_id) + "(size=" + std::to_string(group.size()) + ")";
}

std::string GenNodeDataLabel(const NodeData* node, const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  if (shape_dict.count(node->id())) {
    shape_t node_shape = shape_dict.at(node->id());
    std::stringstream ss;
    ss << node->id() << "\\n{";
    for (size_t i = 0; i < node_shape.size(); ++i) {
      if (i > 0) {
        ss << "x";
      }
      ss << node_shape[i];
    }
    ss << "}";
    return ss.str();
  } else {
    return node->id();
  }
}

std::vector<utils::DotAttr> GetGroupOpAttrs() {
  return std::vector<utils::DotAttr>{
      utils::DotAttr("shape", "Mrecord"), utils::DotAttr("color", "#8EABFF"), utils::DotAttr("style", "filled")};
}

std::vector<utils::DotAttr> GetOutlinkOpAttrs() {
  return std::vector<utils::DotAttr>{
      utils::DotAttr("shape", "Mrecord"), utils::DotAttr("color", "#ff7f00"), utils::DotAttr("style", "filled")};
}

std::vector<utils::DotAttr> GetGroupVarAttrs() {
  return std::vector<utils::DotAttr>{utils::DotAttr("color", "#FFDC85"), utils::DotAttr("style", "filled")};
}

std::vector<utils::DotAttr> GetFetchVarAttrs() {
  return std::vector<utils::DotAttr>{
      utils::DotAttr("peripheries", "2"), utils::DotAttr("color", "#43CD80"), utils::DotAttr("style", "filled")};
}

std::vector<utils::DotAttr> GetGroupAttrs(size_t group_size) {
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
  std::vector<utils::DotAttr> attrs = {
      utils::DotAttr("color", "grey"), utils::DotAttr("style", "filled"), utils::DotAttr("fillcolor", fillcolor)};
  return attrs;
}

void Summary(const std::vector<std::vector<Node*>>& groups, const std::string& viz_path) {
  std::map<std::string, size_t> group_summary;
  std::map<std::string, size_t> single_group_detail;
  std::map<std::string, size_t> fusion_group_detail;

  for (auto& group : groups) {
    size_t group_size = group.size();
    group_summary[std::to_string(group_size)]++;
    if (group_size == 1) {
      // Like "fill_constant_1", remove the "_1" at the end of the string.
      std::string node_id = group[0]->id();
      int index           = node_id.size() - 1;
      while (index != -1) {
        if (node_id[index] >= '0' && node_id[index] <= '9') {
          index--;
        } else {
          break;
        }
      }
      if (node_id[index] == '_') {
        index--;
      }
      if (index >= 0) {
        node_id = node_id.substr(0, index + 1);
        single_group_detail[node_id]++;
      }
    } else {
      std::string key = "others";
      for (auto* node : group) {
        if (node->id().find("reduce") != std::string::npos) {
          key = "reduce";
          break;
        }
      }
      fusion_group_detail[key]++;
    }
  }

  std::stringstream ss;

  auto print_table = [&](const std::map<std::string, size_t>& res) {
    int total = 0;
    for (auto& item : res) {
      ss << std::setw(20) << item.first << item.second << "\n";
      total += item.second;
    }
    ss << "-------------------------------------------\n";
    ss << std::setw(20) << "total" << total << "\n";
    ss << "-------------------------------------------\n";
  };

  ss << "-------------------------------------------\n";
  ss << "             Summary of Groups\n";
  ss << "-------------------------------------------\n";
  ss << std::setiosflags(std::ios::left);
  ss << std::setfill(' ');
  ss << std::setw(20) << "Size"
     << "Numbers\n";
  print_table(group_summary);

  if (single_group_detail.size()) {
    ss << "\n\n-------------------------------------------\n";
    ss << "          Detail of Single Groups\n";
    ss << "-------------------------------------------\n";
    ss << std::setw(20) << "Type"
       << "Numbers\n";
    print_table(single_group_detail);
  }

  ss << "\n\n-------------------------------------------\n";
  ss << "          Detail of Fusion Groups\n";
  ss << "-------------------------------------------\n";
  ss << std::setw(20) << "Type"
     << "Numbers\n";
  print_table(fusion_group_detail);

  std::string filepath = viz_path + "summary.txt";
  WriteToFile(filepath, ss.str());
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

  Summary(groups, viz_path_);

  auto& shape_dict = HasAttr("infershape") ? GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape")
                                           : absl::flat_hash_map<std::string, shape_t>{};

  std::vector<utils::DotAttr> group_op_attrs  = GetGroupOpAttrs();
  std::vector<utils::DotAttr> group_var_attrs = GetGroupVarAttrs();
  std::vector<utils::DotAttr> fetch_var_attrs = GetFetchVarAttrs();

  utils::DotLang dot;
  utils::ResetDotCounters();
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
            std::string label = GenNodeDataLabel(innode, shape_dict);
            dot.AddNode(innode->id(), group_var_attrs, label, cluster_id, true);
            nodedatas_set.insert(innode);
          }
          dot.AddEdge(innode->id(), node->id(), {});
        }
      }
      for (auto& outlink : node->outlinks()) {
        auto* outnode = outlink->sink()->safe_as<NodeData>();
        if (outnode) {
          if (!nodedatas_set.count(outnode)) {
            std::string label = GenNodeDataLabel(outnode, shape_dict);
            if (fetch_var_ids.count(outnode->id())) {
              dot.AddNode(outnode->id(), fetch_var_attrs, label, cluster_id, true);
            } else {
              dot.AddNode(outnode->id(), group_var_attrs, label, cluster_id, true);
            }
            nodedatas_set.insert(outnode);
          }
          dot.AddEdge(node->id(), outnode->id(), {});
        }
      }
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

  std::vector<utils::DotAttr> group_op_attrs  = GetGroupOpAttrs();
  std::vector<utils::DotAttr> group_var_attrs = GetGroupVarAttrs();
  std::vector<utils::DotAttr> fetch_var_attrs = GetFetchVarAttrs();
  std::vector<utils::DotAttr> out_op_attrs    = GetOutlinkOpAttrs();

  utils::ResetDotCounters();

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
          std::string label = GenNodeDataLabel(innode, shape_dict);
          dot.AddNode(innode->id(), group_var_attrs, label, cluster_id, true);
          dot.AddEdge(innode->id(), node->id(), {});
        }
      }
      for (auto& outlink : node->outlinks()) {
        auto* outnode = outlink->sink()->safe_as<NodeData>();
        if (outnode) {
          std::string label = GenNodeDataLabel(outnode, shape_dict);
          if (fetch_var_ids.count(outnode->id())) {
            dot.AddNode(outnode->id(), fetch_var_attrs, label, cluster_id, true);
          } else {
            dot.AddNode(outnode->id(), group_var_attrs, label, cluster_id, true);
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

    std::string filepath = GetFilePathForGroup(groups, group_id, viz_path_);
    WriteToFile(filepath, dot());

    group_id++;
  }
}

std::atomic_size_t Graph::viz_count_{0};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

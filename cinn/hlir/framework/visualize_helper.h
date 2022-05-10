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

#pragma once

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "cinn/hlir/framework/graph.h"
#include "cinn/utils/dot_lang.h"

DECLARE_string(cinn_fusion_groups_graphviz_dir);

namespace cinn {
namespace hlir {
namespace framework {

static bool MakeDirectory(const std::string& dirname, mode_t mode) {
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

static std::string GetFilePathForGroup(const std::vector<std::vector<Node*>>& groups,
                                       const int group_id,
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

static void WriteToFile(const std::string& filepath, const std::string& content) {
  VLOG(4) << "Write to " << filepath;
  std::ofstream of(filepath);
  CHECK(of.is_open()) << "Failed to open " << filepath;
  of << content;
  of.close();
}

static std::string GenClusterId(const std::vector<Node*>& group, int group_id) {
  return "group_" + std::to_string(group_id) + "(size=" + std::to_string(group.size()) + ")";
}

static std::string GenNodeId(const Node* node, bool is_recomputed, int recompute_id) {
  if (is_recomputed) {
    return node->id() + "/" + std::to_string(recompute_id);
  } else {
    return node->id();
  }
}

static std::string GenNodeDataId(const NodeData* data, bool is_recomputed, int recompute_id) {
  if (is_recomputed) {
    return data->id() + "/" + std::to_string(recompute_id);
  } else {
    return data->id();
  }
}

static std::string GenNodeDataLabel(const NodeData* node,
                                    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
                                    const std::string dot_nodedata_id) {
  if (shape_dict.count(node->id())) {
    shape_t node_shape = shape_dict.at(node->id());
    std::stringstream ss;
    ss << dot_nodedata_id << "\\n{";
    for (size_t i = 0; i < node_shape.size(); ++i) {
      if (i > 0) {
        ss << "x";
      }
      ss << node_shape[i];
    }
    ss << "}";
    return ss.str();
  } else {
    return dot_nodedata_id;
  }
}

static std::vector<utils::DotAttr> GetGroupOpAttrs(bool is_recomputed = false) {
  std::string color = is_recomputed ? "#836FFF" : "#8EABFF";
  return std::vector<utils::DotAttr>{
      utils::DotAttr("shape", "Mrecord"), utils::DotAttr("color", color), utils::DotAttr("style", "filled")};
}

static std::vector<utils::DotAttr> GetOutlinkOpAttrs() {
  return std::vector<utils::DotAttr>{
      utils::DotAttr("shape", "Mrecord"), utils::DotAttr("color", "#ff7f00"), utils::DotAttr("style", "filled")};
}

static std::vector<utils::DotAttr> GetGroupVarAttrs(bool is_fetched = false) {
  if (is_fetched) {
    return std::vector<utils::DotAttr>{
        utils::DotAttr("peripheries", "2"), utils::DotAttr("color", "#43CD80"), utils::DotAttr("style", "filled")};
  } else {
    return std::vector<utils::DotAttr>{utils::DotAttr("color", "#FFDC85"), utils::DotAttr("style", "filled")};
  }
}

static std::vector<utils::DotAttr> GetGroupAttrs(size_t group_size) {
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

static void Summary(const std::vector<std::vector<Node*>>& groups, const std::string& viz_path) {
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

static std::string DebugString(const Node* node) {
  std::stringstream ss;
  ss << "op_type=" << node->id() << ", inputs{";
  bool first = true;
  for (auto& inlink : node->inlinks()) {
    auto* innode = inlink->source()->safe_as<NodeData>();
    if (innode) {
      if (!first) {
        ss << ", ";
      } else {
        first = false;
      }
      ss << innode->id();
    }
  }
  ss << "}, outputs{";
  first = true;
  for (auto& outlink : node->outlinks()) {
    auto* outnode = outlink->sink()->safe_as<NodeData>();
    if (outnode) {
      if (!first) {
        ss << ", ";
      } else {
        first = false;
      }
      ss << outnode->id();
    }
  }
  ss << "}";
  return ss.str();
}

static void FindRecomputeNodes(const std::vector<std::vector<Node*>>& groups,
                               std::unordered_map<std::string, int>* recompute_nodes) {
  std::unordered_map<std::string, int> op_count;
  for (auto& group : groups) {
    for (auto* node : group) {
      op_count[node->id()]++;
    }
  }
  for (auto& iter : op_count) {
    if (iter.second > 1) {
      (*recompute_nodes)[iter.first] = 0;
    }
  }
}

static void AddGroupNode(const Node* node,
                         const std::string& dot_cluster_id,
                         const std::unordered_set<std::string>& fetch_var_ids,
                         const absl::flat_hash_map<std::string, shape_t>& shape_dict,
                         std::unordered_map<std::string, int>* recompute_nodes,
                         std::unordered_map<std::string, std::string>* outnode2dot_id,
                         std::unordered_set<std::string>* nodedatas_set,
                         utils::DotLang* dot) {
  bool is_recomputed = recompute_nodes->count(node->id());
  int recompute_id   = is_recomputed ? (*recompute_nodes)[node->id()]++ : -1;

  std::string dot_node_id = GenNodeId(node, is_recomputed, recompute_id);
  dot->AddNode(dot_node_id, GetGroupOpAttrs(is_recomputed), "", dot_cluster_id);

  for (auto& inlink : node->inlinks()) {
    auto* innode = inlink->source()->safe_as<NodeData>();
    if (innode) {
      if (!outnode2dot_id->count(innode->id())) {
        (*outnode2dot_id)[innode->id()] = innode->id();
      }
      std::string dot_innode_id = outnode2dot_id->at(innode->id());
      if (!nodedatas_set || !nodedatas_set->count(dot_innode_id)) {
        std::string label = GenNodeDataLabel(innode, shape_dict, dot_innode_id);
        dot->AddNode(dot_innode_id, GetGroupVarAttrs(false), label, dot_cluster_id, true);
        if (nodedatas_set) {
          nodedatas_set->insert(dot_innode_id);
        }
      }
      dot->AddEdge(dot_innode_id, dot_node_id, {});
    }
  }

  for (auto& outlink : node->outlinks()) {
    auto* outnode = outlink->sink()->safe_as<NodeData>();
    if (outnode) {
      std::string dot_outnode_id       = GenNodeDataId(outnode, is_recomputed, recompute_id);
      (*outnode2dot_id)[outnode->id()] = dot_outnode_id;
      if (!nodedatas_set || !nodedatas_set->count(dot_outnode_id)) {
        bool is_fetched   = fetch_var_ids.count(outnode->id());
        std::string label = GenNodeDataLabel(outnode, shape_dict, dot_outnode_id);
        dot->AddNode(dot_outnode_id, GetGroupVarAttrs(is_fetched), label, dot_cluster_id, true);
        if (nodedatas_set) {
          nodedatas_set->insert(dot_outnode_id);
        }
      }
      dot->AddEdge(dot_node_id, dot_outnode_id, {});
    }
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

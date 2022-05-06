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

#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/types/any.h>

#include <memory>
#include <string>
#include <vector>

#include "cinn/common/graph_utils.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * \brief Symbolic computation graph.
 *  This is the intermediate representation for optimization pass.
 */
class Graph : public cinn::common::Graph {
 public:
  Graph(const frontend::Program& prog, const Target& target);

  Target target_;
  /** \brief outputs of the computation graph. */
  std::vector<NodeData*> outputs;

  /** \brief attributes of a graph */
  absl::flat_hash_map<std::string, std::shared_ptr<absl::any>> attrs;

  std::vector<std::vector<Node*>> groups;
  struct Group {
    std::string group_id{""};
    // node in this group
    std::vector<Node*> nodes;
    std::unordered_set<Node*> nodes_set;
    // input nodes of the group.
    std::unordered_map<Node*, int> input_nodes;
    // output nodes of the group.
    std::unordered_set<Node*> output_nodes;
    // op pattern kind.
    framework::OpPatternKind op_pattern_kind{framework::kElemWise};
    // internal node, the output is used by multi-node.
    // internal node can't use compute inline, should use buffer.
    std::unordered_set<Node*> internal_nodes;
    // master node for schedule
    std::unordered_set<Node*> master_nodes;

    struct SharedGroupHasher {
      size_t operator()(const std::shared_ptr<Group>& group) const noexcept {
        return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(group.get()));
      }
    };
    struct SharedGroupComparator {
      bool operator()(const std::shared_ptr<Group>& first, const std::shared_ptr<Group>& second) const noexcept {
        return first.get() == second.get();
      }
    };
    // input groups
    std::unordered_set<std::shared_ptr<Group>, SharedGroupHasher, SharedGroupComparator> producer_groups;
    // output grous
    std::unordered_set<std::shared_ptr<Group>, SharedGroupHasher, SharedGroupComparator> consumer_groups;
    // fused sub-groups, used for fusion merge pass
    std::vector<std::shared_ptr<Group>> fused_sub_groups;
    // if as sub-group, used for belong groups.
    std::unordered_set<std::shared_ptr<Group>, SharedGroupHasher, SharedGroupComparator> belong_groups;

    // for op lowering.
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    std::vector<Node*> CollectNodes() {
      if (fused_sub_groups.size()) {
        std::vector<Node*> tmp_nodes;
        for (auto& group : fused_sub_groups) {
          tmp_nodes.insert(tmp_nodes.end(), group->nodes.begin(), group->nodes.end());
        }
        return tmp_nodes;
      } else {
        return nodes;
      }
    }

    std::string GetFuncName() { return "fn_" + group_id; }
  };
  std::vector<std::shared_ptr<Group>> fusion_groups;

  void RegisterNode(size_t key, Node* node) { this->common::Graph::RegisterNode(key, node->as<common::GraphNode>()); }
  void RegisterNode(size_t key, NodeData* node) {
    this->common::Graph::RegisterNode(key, node->as<common::GraphNode>());
  }
  void RegisterNode(const std::string& key, Node* node) {
    this->common::Graph::RegisterNode(key, node->as<common::GraphNode>());
  }
  void RegisterNode(const std::string& key, NodeData* node) {
    this->common::Graph::RegisterNode(key, node->as<common::GraphNode>());
  }

  /**
   * \brief Get the immutable attribute from attrs.
   * @param attr_name the name of the attribute
   * @return the reference to corresponding attribute
   * @tparam T the type of the attribute.
   */
  template <typename T>
  inline const T& GetAttrs(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    CHECK(it != attrs.end()) << "Cannot find attribute [" << attr_name << "] in the graph";
    return absl::any_cast<const T&>(*it->second);
  }

  /**
   * \brief Get the mutable attribute from attrs.
   * @param attr_name the name of the attribute
   * @return the reference to corresponding attribute
   * @tparam T the type of the attribute.
   */
  template <typename T>
  inline T& GetMutableAttrs(const std::string& attr_name) {
    auto it = attrs.find(attr_name);
    CHECK(it != attrs.end()) << "Cannot find attribute [" << attr_name << "] in the graph";
    return absl::any_cast<T&>(*it->second);
  }

  /**
   * \brief Check whether has a specific attribute.
   * @param attr_name the name of the attribute
   * @return a boolean result
   */
  inline bool HasAttr(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    return it != attrs.end();
  }

  void VisualizeGroupedGraph(const std::vector<std::vector<Node*>>& groups,
                             const std::unordered_set<std::string>& fetch_var_ids);

 private:
  void VisualizeGroups(const std::vector<std::vector<Node*>>& groups,
                       const std::unordered_set<std::string>& fetch_var_ids);

  CINN_DISALLOW_COPY_AND_ASSIGN(Graph);
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

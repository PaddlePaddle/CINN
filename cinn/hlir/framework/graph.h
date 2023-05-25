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

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "cinn/common/graph_utils.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/common/hashable_weak_ptr.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * \brief Symbolic computation graph.
 *  This is the intermediate representation for optimization pass.
 */
class Graph : public cinn::common::Graph {
 public:
  Graph(const frontend::Program& prog, const Target& target) {
    std::unordered_set<std::string> fetch_var_ids;
    Initialize(prog, fetch_var_ids, target);
  }
  Graph(const frontend::Program& prog, const std::unordered_set<std::string>& fetch_var_ids, const Target& target) {
    Initialize(prog, fetch_var_ids, target);
  }

  void Initialize(const frontend::Program& prog,
                  const std::unordered_set<std::string>& fetch_var_ids,
                  const Target& target);

  Target target_;
  /** \brief outputs of the computation graph. */
  std::vector<NodeData*> outputs;

  /** \brief attributes of a graph */
  absl::flat_hash_map<std::string, std::shared_ptr<absl::any>> attrs;

  std::vector<std::vector<Node*>> groups;

  struct Group : public std::enable_shared_from_this<Group> {
   public:
    // distance to last group.
    int depth{0};
    int max_depth{0};
    int min_depth{INT_MAX};
    // group id, consisted of node's id.
    std::string group_id{""};
    // global unique id.
    std::string unique_id{UniqName("")};
    // node in this group
    std::vector<Node*> nodes;
    std::unordered_set<Node*> nodes_set;
    // input nodes of the group.
    std::unordered_map<Node*, int> input_nodes;
    // output nodes of the group.
    std::unordered_set<Node*> output_nodes;
    // op pattern kind.
    framework::OpPatternKind op_pattern_kind{framework::kElementWise};
    // internal node, the output is used by multi-node.
    // internal node can't use compute inline, should use buffer.
    std::unordered_set<Node*> internal_nodes;
    // master node for schedule
    std::unordered_set<Node*> master_nodes;

    // for op lowering.
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    // Getters
    // input groups
    const std::unordered_set<HashableWeakPtr<Group>>& producer_groups() const {
      return producer_groups_;
    }
    // output grous
    const std::unordered_set<HashableWeakPtr<Group>>& consumer_groups() const {
      return consumer_groups_;
    }
    // fused sub-groups, used for fusion merge pass
    const std::vector<HashableWeakPtr<Group>>& fused_sub_groups() const {
      return fused_sub_groups_;
    }
    // if as sub-group, used for belong groups.
    const std::unordered_set<HashableWeakPtr<Group>>& belong_groups() const {
      return belong_groups_;
    }

    void ConnectTo(const std::weak_ptr<Group>& consumer) {
      consumer_groups_.insert(consumer);
      consumer->producer_groups_.insert(shared_from_this());
    }

    void DisConnectTo(const std::weak_ptr<Group>& consumer) {
      consumer_groups_.erase(consumer);
      consumer->producer_groups_.erase(shared_from_this());
      // TODO: delete `shared_from_this()` or `consumer` from `owner` if they isolated and unused.
    }

    void AddSubGroup(const std::weak_ptr<Group>& sub_group) {
      fused_sub_groups_.push_back(sub_group);
      sub_group.belong_groups_.insert(shared_from_this());
    }

    void RemoveSubGroup(const std::weak_ptr<Group>& sub_group) {
      const auto& pos = std::find(fused_sub_groups_.begin(), fused_sub_groups_.end(), sub_group);
      if (pos == fused_sub_groups_.end()) {
        CHECK_EQ(sub_group.belong_groups_.count(shared_from_this()), 0);
      } else {
        fused_sub_groups_.erase(pos);
        sub_group.belong_groups_.erase(shared_from_this());
      }
      // TODO: delete `shared_from_this()` or `consumer` from `owner` if they isolated and unused.
    }

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

    std::unordered_set<Node*> NodeSet() {
      std::unordered_set<Node*> node_set;
      for (auto node : CollectNodes()) {
        node_set.insert(node);
      }
      return node_set;
    }

    std::unordered_set<NodeData*> GetInputNodeDatas();
    std::unordered_set<NodeData*> GetOutputNodeDatas();

    std::string GetFuncName() { return "fn_" + group_id + unique_id; }

   private:
    friend class Graph;
    Group(const std::weak_ptr<std::unordered_set<std::shared_ptr<Group>>>& owner) : owner_(owner) {}

    // input groups
    std::unordered_set<HashableWeakPtr<Group>> producer_groups_;
    // output grous
    std::unordered_set<HashableWeakPtr<Group>> consumer_groups_;
    // fused sub-groups, used for fusion merge pass
    std::vector<std::weak_ptr<Group>> fused_sub_groups_;
    // if as sub-group, used for belong groups.
    std::unordered_set<HashableWeakPtr<Group>> belong_groups_;
    std::weak_ptr<std::unordered_set<std::shared_ptr<Group>>> owner_;
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

  /**
   * \brief Debug the grouped graph according to fusion_groups.
   */
  std::string DebugGroupedGraph(const std::unordered_set<std::string>& fetch_var_ids = {});
  std::string DebugGroupedGraph(const std::vector<Node*>& group,
                                const std::unordered_set<std::string>& fetch_var_ids = {});

  /**
   * \brief Debug the grouped graph with GraphViz dot format according to fusion_groups.
   */
  std::string VisualizeGraph(const std::unordered_set<std::string>& fetch_var_ids = {});
  std::vector<std::string> VisualizeGroups(const std::unordered_set<std::string>& fetch_var_ids = {});

  /**
   * \brief Genereate the python test code for group test
   */
  std::string GenerateGroupPythonCode(const std::vector<Node*>& group,
                                      const std::unordered_set<std::string>& fetch_var_ids = {});

  /**
   * \brief Visualize the grouped graph according to fusion_groups.
   */
  void VisualizeGroupedGraph(const std::unordered_set<std::string>& fetch_var_ids = {});

  /**
   * \brief Visualize the grouped graph according to user specified groups.
   */
  void VisualizeGroupedGraph(const std::vector<std::vector<Node*>>& groups,
                             const std::unordered_set<std::string>& fetch_var_ids = {});

  void SaveSourceCode(const std::string& code);
  void SavePTXCode(const std::string& ptx);

  std::weak_ptr<Group> CreateGroup();

 private:
  std::string DebugGroupedGraph(const std::vector<std::vector<Node*>>& groups,
                                const std::unordered_set<std::string>& fetch_var_ids = {});

  std::string VisualizeGraph(const std::vector<std::vector<Node*>>& groups,
                             const std::unordered_set<std::string>& fetch_var_ids = {});

  std::vector<std::string> VisualizeGroups(const std::vector<std::vector<Node*>>& groups,
                                           const std::unordered_set<std::string>& fetch_var_ids = {});

  std::vector<std::vector<Node*>> FusionGroupsToGroups();

  std::string viz_path_;
  static std::atomic_size_t viz_count_;

  CINN_DISALLOW_COPY_AND_ASSIGN(Graph);

  // For groups ownership.
  std::shared_ptr<std::unordered_set<std::shared_ptr<Group>>> groups_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

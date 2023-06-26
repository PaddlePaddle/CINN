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

#include <memory>

#include "cinn/api/op_node.h"

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace api {

using Comparator = hlir::framework::Graph::Group::SharedGroupComparator;
using Hasher     = hlir::framework::Graph::Group::SharedGroupHasher;

class OpGroup {
 public:
  OpGroup(const std::shared_ptr<hlir::framework::Graph::Group>& group, const hlir::framework::Graph* graph) : group_(group), graph_(graph) {}

  OpGroup(const OpGroup& other) = default;

  class OpNodeListView {
   public:
    explicit OpNodeListView(std::vector<hlir::framework::Node*> op_nodes, const cinn::hlir::framework::Graph* graph) : op_nodes_(std::move(op_nodes)), graph_(graph) {}

    OpNodeListView(const OpNodeListView& other) = delete;
    OpNodeListView(OpNodeListView&& other) = delete;

    OpNodeListView& operator=(const OpNodeListView& other) = delete;

    class Iterator {
     public:
      Iterator(std::vector<hlir::framework::Node*>::const_iterator it, const hlir::framework::Graph* graph) : iter_(it), graph_(graph) {}

      Iterator& operator++() {
        ++iter_;
        return *this;
      }

      Iterator operator++(int) {
        Iterator tmp = *this;
        ++iter_;
        return tmp;
      }

      bool operator==(const Iterator& other) const {
        return iter_ == other.iter_;
      }

      bool operator!=(const Iterator& other) const {
          return !(*this == other);
      }

      OpNode operator*() const {
        return OpNode(*iter_, graph_);
      }

     private:
      std::vector<hlir::framework::Node*>::const_iterator iter_;
      const hlir::framework::Graph* graph_;
    };

    size_t size() const { return op_nodes_.size(); }

    Iterator begin() { return Iterator(op_nodes_.begin(), graph_); }

    Iterator end() { return Iterator(op_nodes_.begin(), graph_); }
   private:
    std::vector<hlir::framework::Node*> op_nodes_;
    const cinn::hlir::framework::Graph* graph_;
  };

  class OpGroupListView {
   public:
    OpGroupListView(const std::unordered_map<std::shared_ptr<hlir::framework::Graph::Group>, TensorInterfaceList, Hasher, Comparator>& group_map, const hlir::framework::Graph* graph) : op_group_map_(group_map), graph_(graph) {}

    OpGroupListView(const OpGroupListView& other) = delete;
    OpGroupListView(OpGroupListView&& other) = delete;

    OpGroupListView& operator=(const OpGroupListView& other) = delete;

    class Iterator {
     public:
      Iterator(std::unordered_map<std::shared_ptr<hlir::framework::Graph::Group>, TensorInterfaceList, Hasher, Comparator>::const_iterator it, const hlir::framework::Graph* graph) : iter_(it), graph_(graph) {}

      Iterator& operator++() {
        ++iter_;
        return *this;
      }

      Iterator operator++(int) {
        Iterator tmp = *this;
        ++iter_;
        return tmp;
      }

      bool operator==(const Iterator& other) const {
        return iter_ == other.iter_;
      }

      bool operator!=(const Iterator& other) const {
          return !(*this == other);
      }

      OpGroup operator*() const{
        return OpGroup(iter_->first, graph_);
      }

     private:
      std::unordered_map<std::shared_ptr<hlir::framework::Graph::Group>, TensorInterfaceList, Hasher, Comparator>::const_iterator iter_;
      const hlir::framework::Graph* graph_;
    };

    size_t size() const { return op_group_map_.size(); }

    Iterator begin() { return Iterator(op_group_map_.begin(), graph_); }

    Iterator end() { return Iterator(op_group_map_.begin(), graph_); }

   private:
    const std::unordered_map<std::shared_ptr<hlir::framework::Graph::Group>, TensorInterfaceList, Hasher, Comparator>& op_group_map_;
    const cinn::hlir::framework::Graph* graph_;
  };



  hlir::framework::OpPatternKind kind() const { return group_->kind(); }

  OpNodeListView ops() const {
    return OpNodeListView(group_->CollectNodes(), graph_);
  }

  OpGroupListView Producers() const {
    return OpGroupListView(group_->producer_groups(), graph_);
  }

  OpGroupListView Consumers() const {
    return OpGroupListView(group_->consumer_groups(), graph_);
  }

  std::shared_ptr<hlir::framework::Graph::Group> GetGroup() const {
    return group_;
  }

  bool operator == (const OpGroup& other) const {
    return group_.get() == other.group_.get();
  }

  bool operator < (const OpGroup& other) const {
    return group_.get() < other.group_.get();
  }

 private:
  const std::shared_ptr<hlir::framework::Graph::Group> group_;
  const hlir::framework::Graph* graph_;
};

}  // namespace api
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::api::OpGroup> {
  size_t operator()(const cinn::api::OpGroup& obj) const {
    return std::hash<int64_t>()(reinterpret_cast<uint64_t>(obj.GetGroup().get()));
  }
};

} // namespace std

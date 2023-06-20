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

class OpGroup {
 public:
  OpGroup(const hlir::pass::FusionHelperBase* helper, const std::shared_ptr<hlir::framework::Graph::Group>& group) : helper_(helper), group_(group) {}

  OpGroup(const OpGroup& other) = default;

  class iterator {
   public:
    iterator(std::unordered_map<std::shared_ptr<hlir::framework::Graph::Group>, TensorInterfaceList>::iterator it, const hlir::pass::FusionHelperBase* helper) : iter_(it), helper_(helper) {}

    iterator& operator++() {
      ++iter_;
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++iter_;
      return tmp;
    }

    std::shared_ptr<OpGroup> operator*() {
      return std::make_shared<OpGroup>(helper_, iter_->first);
    }

    bool operator==(const iterator& other) const {
      return iter_ == other.iter_;
    }

    bool operator!=(const iterator& other) const {
        return !(*this == other);
    }

   private:
    std::unordered_map<std::shared_ptr<hlir::framework::Graph::Group>, TensorInterfaceList>::iterator iter_;
    const hlir::pass::FusionHelperBase* helper_;
  };

  hlir::framework::OpPatternKind kind() const { return group_->kind(); }

  size_t OpSize() const {
    return group_->CollectNodes().size();
  }

  OpNode GetOp(size_t index) const {
    return OpNode(helper_, group_->CollectNodes()[index]);
  }

  size_t ProducerSize() const {
    return group_->producer_groups().size();
  }

  size_t ConsumerSize() const {
    return group_->consumer_groups().size();
  }

  iterator ProducerBegin() const {
    return iterator(group_->mut_producer_groups()->begin(), helper_);
  }

  iterator ProducerEnd() const {
    return iterator(group_->mut_producer_groups()->end(), helper_);
  }

  iterator ConsumerBegin() const {
    return iterator(group_->mut_consumer_groups()->begin(), helper_);
  }

  iterator ConsumerEnd() const {
    return iterator(group_->mut_consumer_groups()->end(), helper_);
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
  const hlir::pass::FusionHelperBase* helper_;
  const std::shared_ptr<hlir::framework::Graph::Group> group_;
};

}  // namespace api
}  // namespace cinn

namespace std {
  template <>
  struct hash<MyClass> {
    size_t operator()(const cinn::api::OpGroup& obj) const {
      return std::hash<int64_t>{}(obj.GetGroup().get());
    }
  };
}

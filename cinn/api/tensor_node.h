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

#include "cinn/hlir/framework/graph.h"
#include "cinn/utils/type_defs.h"
#include "cinn/hlir/pass/fusion_helper_base.h"
#include "cinn/utils/small_vector.h"

namespace cinn {
namespace api {

using shape_t = utils::ShapeType;

class OpNode;

class Shape final {
 public:
  explicit Shape(const utils::ShapeType& shape) : shape_(shape.begin(), shape.end()) {}

  Shape(const Shape& other) = delete;
  Shape(Shape&& other) = delete;

  Shape& operator=(const Shape& other) = delete;

  bool operator == (const Shape& other) const {
    if (shape_.size() != other.shape_.size()) {
        return false;
    }
    return std::equal(shape_.begin(), shape_.end(), other.shape_.begin());
  }

  size_t operator[] (size_t index) const {
    return shape_[index];
  }

  size_t at(size_t index) const {
    return shape_[index];
  }

  size_t size() const {
    return shape_.size();
  }

  // Returns the total number of elements in the shape.
  size_t numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());;
  }

 private:
  cinn::utils::SmallVector<int64_t, 12> shape_;
};


class TensorNode final {
 public:
  TensorNode(const hlir::framework::NodeData* node_data, const hlir::framework::Graph* graph) : node_data_(node_data), graph_(graph), consumers_(node_data_->outlinks(), graph_) {
    const auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
    CHECK(shape_dict.count(node_data_->id())) << "Can't find " << node_data_->id() << " 's shape!";
    shape_ = std::make_shared<Shape>(shape_dict.find(node_data_->id())->second);
  }

  // Get the shape of tensor.
  const Shape& shape() const {
    return *shape_;
  }

  // Input data has no producer.
  bool HasProducer() const {
    return node_data_->source_node.get() != nullptr;
  }

  OpNode producer() const;

  class ConsumerOpListView {
   public:
    ConsumerOpListView(const std::set<common::Shared<common::GraphEdge>, common::GraphEdgeCompare>& edges, const hlir::framework::Graph* graph) : edges_(edges), graph_(graph) {}

    ConsumerOpListView(const ConsumerOpListView& other) = delete;
    ConsumerOpListView(ConsumerOpListView&& other) = delete;

    ConsumerOpListView& operator=(const ConsumerOpListView& other) = delete;

    class Iterator {
     public:
      Iterator(std::set<common::Shared<common::GraphEdge>, common::GraphEdgeCompare>::const_iterator it, const hlir::framework::Graph* graph) : iter_(it), graph_(graph) {}

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

      OpNode operator*() const;

     private:
      std::set<common::Shared<common::GraphEdge>, common::GraphEdgeCompare>::const_iterator iter_;
      const hlir::framework::Graph* graph_;
    };

    size_t size() const { return edges_.size(); }

    Iterator begin() const {
      return  Iterator(this->edges_.begin(), graph_);
    }

    Iterator end() const {
      return  Iterator(this->edges_.end(), graph_);
    }

   private:
    const std::set<Shared<common::GraphEdge>, common::GraphEdgeCompare>& edges_;
    const hlir::framework::Graph* graph_;
  };

  const ConsumerOpListView& consumers() const {
    return consumers_;
  }

 private:
  const hlir::framework::NodeData* node_data_;
  const hlir::framework::Graph* graph_;

  std::shared_ptr<Shape> shape_;
  const ConsumerOpListView consumers_;
};

}  // namespace api
}  // namespace cinn

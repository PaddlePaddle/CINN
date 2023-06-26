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
#include "cinn/hlir/pass/fusion_helper_base.h"
#include "cinn/api/tensor_node.h"
#include "cinn/hlir/framework/op.h"

namespace cinn {
namespace api {

using OpPatternKind = cinn::hlir::framework::OpPatternKind;
using Attribute = cinn::utils::Attribute;

class OpNode {
 public:
  OpNode(const hlir::framework::Node* node, const hlir::framework::Graph* graph) : node_(node), graph_(graph) {
    input_edges_ = node->inlinks_in_order();
    output_edges_ = node->outlinks_in_order();
  }

  OpPatternKind kind () const {
    thread_local const static hlir::framework::OpValueType<OpPatternKind>& op_pattern_dict = hlir::framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
    auto kind = op_pattern_dict[node_->op()];

    if (kind == hlir::framework::kBroadcast) {
      // As binary op was defined as broadcast, actually it should be element-wise.
      if (node_->op()->name != "broadcast_to") {
        return hlir::framework::kElementWise;
      }
    }
    return kind;
  }

  class InputTensorListView {
   public:
    InputTensorListView(const hlir::framework::Graph* graph, const std::vector<common::Shared<common::GraphEdge>>& edges) : graph_(graph), edges_(edges) {}

    InputTensorListView(const InputTensorListView& other) = delete;
    InputTensorListView(InputTensorListView&& other) = delete;

    InputTensorListView& operator=(const InputTensorListView& other) = delete;

    size_t size() const { return edges_.size(); }

    TensorNode operator[](size_t index) const;

   private:
    const hlir::framework::Graph* graph_;
    const std::vector<common::Shared<common::GraphEdge>>& edges_;
  };

  class OutputTensorListView {
   public:
    OutputTensorListView(const hlir::framework::Graph* graph, const std::vector<common::Shared<common::GraphEdge>>& edges) : graph_(graph), edges_(edges) {}

    OutputTensorListView(const OutputTensorListView& other) = delete;

    OutputTensorListView(OutputTensorListView&& other) = delete;

    size_t size() const { return edges_.size(); }

    TensorNode operator[](size_t index) const;

   private:
    const hlir::framework::Graph* graph_;
    const std::vector<common::Shared<common::GraphEdge>>& edges_;
  };

  size_t InputsSize() const {
    return node_->inlinks().size();
  }

  size_t OutputsSize() const {
    return node_->outlinks().size();
  }

  InputTensorListView Inputs() const {
    return InputTensorListView(graph_, input_edges_);
  }

  OutputTensorListView Outputs() const {
    return OutputTensorListView(graph_, output_edges_);
  }

  template <typename T>
  const T& GetAttr(const std::string& attr_name) const {
    return absl::get<T>(GetAttr(attr_name));
  }

 private:
  const Attribute& GetAttr(const std::string& attr_name) const {
    return node_->attrs.attr_store.at(attr_name);
  }

  const hlir::framework::Node* node_;
  const hlir::framework::Graph* graph_;

  std::vector<common::Shared<common::GraphEdge>> input_edges_;
  std::vector<common::Shared<common::GraphEdge>> output_edges_;
};

}  // namespace api
}  // namespace cinn

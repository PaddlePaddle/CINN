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

#include <queue>

#include "cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace hlir {
namespace pass {

inline Expr GetScalarExpr(const framework::NodeAttr::attr_t& attr) {
  Expr scalar;
  struct Visitor {
    Expr& scalar_;
    explicit Visitor(Expr& scalar) : scalar_(scalar) {}
    void operator()(float v) { scalar_ = Expr(v); }
    void operator()(double v) { scalar_ = Expr(v); }
    void operator()(int32_t v) { scalar_ = Expr(v); }
    void operator()(int64_t v) { scalar_ = Expr(v); }
    void operator()(bool v) { scalar_ = Expr(v); }
    void operator()(const std::string& v) { scalar_ = Expr(v); }
    void operator()(const std::vector<int>&) { LOG(FATAL) << "wrong type std::vector<int>"; }
    void operator()(const std::vector<int64_t>&) { LOG(FATAL) << "wrong type std::vector<int64_t>"; }
    void operator()(const std::vector<float>&) { LOG(FATAL) << "wrong type std::vector<float>"; }
    void operator()(const std::vector<double>&) { LOG(FATAL) << "wrong type std::vector<double>"; }
    void operator()(const std::vector<bool>&) { LOG(FATAL) << "wrong type std::vector<bool>"; }
    void operator()(const std::vector<std::string>&) { LOG(FATAL) << "wrong type std::vector<std::string>"; }
  };
  absl::visit(Visitor{scalar}, attr);
  return scalar;
}

inline void fold_broadcast_to_constant(const FusionHelperBase* helper, Graph* graph, Node* node) {
  auto scalar_op = helper->GetProducerNode(node)[0];
  CHECK(node->attrs.attr_store.count("out_shape"));
  auto shape = std::get<std::vector<int>>(node->attrs.attr_store.at("out_shape"));
  CHECK(scalar_op->attrs.attr_store.count("value"));
  auto scalar = GetScalarExpr(scalar_op->attrs.attr_store.at("value"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"), "fill_constant", common::UniqName("fill_constant"));
  // set node attr
  node_tmp->attrs.attr_store["shape"]     = shape;
  node_tmp->attrs.attr_store["value"]     = scalar;
  node_tmp->attrs.attr_store["force_cpu"] = false;
  this->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto scalar_node_data = helper->GetNodeData(scalar_op);
  if (scalar_node_data->outlinks().size() == 1) {
    graph->DropNode(node_tmp);
    graph->DropNode(scalar_op);
    graph->DropNode(scalar_node_data);
  } else {
    scalar_node_data->UnLinkSingleTo(node);
    graph->DropNode(node_tmp);
  }
}

inline void fold_reshape_fill_constant(const FusionHelperBase* helper, Graph* graph, Node* node) {
  auto scalar_op = helper->GetProducerNode(node)[0];
  CHECK(node->attrs.attr_store.count("shape"));
  auto shape = std::get<std::vector<int>>(node->attrs.attr_store.at("shape"));
  CHECK(scalar_op->attrs.attr_store.count("value"));
  auto scalar = GetScalarExpr(scalar_op->attrs.attr_store.at("value"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"), "fill_constant", common::UniqName("fill_constant"));
  // set node attr
  node_tmp->attrs.attr_store["shape"]     = shape;
  node_tmp->attrs.attr_store["value"]     = scalar;
  node_tmp->attrs.attr_store["force_cpu"] = false;
  this->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto scalar_node_data = helper->GetNodeData(scalar_op);
  if (scalar_node_data->outlinks().size() == 1) {
    graph->DropNode(node_tmp);
    graph->DropNode(scalar_op);
    graph->DropNode(scalar_node_data);
  } else {
    scalar_node_data->UnLinkSingleTo(node);
    graph->DropNode(node_tmp);
  }
}

inline void fold_squeeze_fill_constant(const FusionHelperBase* helper, Graph* graph, Node* node) {
  auto scalar_op = helper->GetProducerNode(node)[0];
  CHECK(node->attrs.attr_store.count("shape"));
  auto axes = std::get<std::vector<int>>(node->attrs.attr_store.at("shape"));
  CHECK(scalar_op->attrs.attr_store.count("value"));
  auto scalar = GetScalarExpr(scalar_op->attrs.attr_store.at("value"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"), "fill_constant", common::UniqName("fill_constant"));
  // set node attr
  node_tmp->attrs.attr_store["shape"]     = shape;
  node_tmp->attrs.attr_store["value"]     = scalar;
  node_tmp->attrs.attr_store["force_cpu"] = false;
  this->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto scalar_node_data = helper->GetNodeData(scalar_op);
  if (scalar_node_data->outlinks().size() == 1) {
    graph->DropNode(node_tmp);
    graph->DropNode(scalar_op);
    graph->DropNode(scalar_node_data);
  } else {
    scalar_node_data->UnLinkSingleTo(node);
    graph->DropNode(node_tmp);
  }
}

inline void fold_expand_dims_to_fill_constant(const FusionHelperBase* helper, Graph* graph, Node* node) {
  auto scalar_op = helper->GetProducerNode(node)[0];
  CHECK(node->attrs.attr_store.count("shape"));
  auto shape = std::get<std::vector<int>>(node->attrs.attr_store.at("shape"));
  CHECK(scalar_op->attrs.attr_store.count("value"));
  auto scalar = GetScalarExpr(scalar_op->attrs.attr_store.at("value"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"), "fill_constant", common::UniqName("fill_constant"));
  // set node attr
  node_tmp->attrs.attr_store["shape"]     = shape;
  node_tmp->attrs.attr_store["value"]     = scalar;
  node_tmp->attrs.attr_store["force_cpu"] = false;
  this->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto scalar_node_data = helper->GetNodeData(scalar_op);
  if (scalar_node_data->outlinks().size() == 1) {
    graph->DropNode(node_tmp);
    graph->DropNode(scalar_op);
    graph->DropNode(scalar_node_data);
  } else {
    scalar_node_data->UnLinkSingleTo(node);
    graph->DropNode(node_tmp);
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

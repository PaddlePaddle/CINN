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

#include "cinn/hlir/framework/op_lowering.h"

namespace cinn {
namespace hlir {
namespace framework {

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::OpPatternKind;
using framework::shape_t;
using framework::StrategyFunction;

using common::GraphEdge;
using common::GraphNode;
using common::Type;
using namespace lang;

NodeData* GetNodeData(Node* node) {
  auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
  CHECK(node_data);
  return node_data;
}

std::vector<Node*> GetConsumer(Node* node) {
  std::vector<Node*> consumers;
  auto node_data = GetNodeData(node);
  for (auto& link : node_data->outlinks()) {
    auto consumer_node = link->sink()->safe_as<Node>();
    CHECK(consumer_node);
    consumers.push_back(consumer_node);
  }
  return consumers;
}

OpLoweringHelper::OpLoweringHelper(const std::unordered_set<std::string>& fetch_var_ids,
                                   const absl::flat_hash_map<std::string, Type>& type_dict,
                                   const absl::flat_hash_map<std::string, shape_t>& shape_dict,
                                   const Target& target)
    : fetch_var_ids_(fetch_var_ids), type_dict_(type_dict), shape_dict_(shape_dict), target_(target) {}

/*
ir::LoweredFunc Lower(const std::string &name,
                      StageMap stages,
                      const std::vector<Tensor> &tensor_args,
                      const std::vector<Var> &scalar_args     = {},
                      const std::vector<Tensor> &temp_tensors = {},
                      ir::Module::Builder *b                  = nullptr,
                      const Target &target                    = common::DefaultHostTarget());
*/

// elementwise fusion op lowering
ir::LoweredFunc OpLoweringHelper::ElementwiseOpLowering(const Group& group) {
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  // get input tensor and output tensor
  poly::StageMap stages;
  std::string func_name = "fn_fuse";
  std::vector<ir::Tensor> func_args;
  std::unordered_map<std::string, ir::Tensor> tensor_map;

  auto compute = [this, &strategy, &group, &stages, &func_name, &func_args, &tensor_map](const Group& sub_group) {
    // compute
    for (int idx = sub_group->nodes_.size() - 1; idx >= 0; --idx) {
      auto node = sub_group->nodes_[idx];
      func_name += "_" + node->id();

      std::vector<ir::Tensor> tensor_inputs;
      std::vector<common::CINNValue> cinn_inputs;
      // get all input nodes
      for (auto& link : node->inlinks_in_order(true)) {
        auto source = link->source();
        CHECK(source);
        auto source_data = source->safe_as<NodeData>();
        CHECK(source_data);

        if (tensor_map.count(source_data->id())) {
          tensor_inputs.push_back(tensor_map[source_data->id()]);
          cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor_map[source_data->id()])));
        } else {
          auto tensor = lang::Placeholder<float>(source_data->id(), this->shape_dict_.at(source_data->id()));
          tensor_map[source_data->id()] = tensor;

          tensor_inputs.push_back(tensor);
          cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
          // recored func input args
          func_args.push_back(tensor);
        }
      }

      std::vector<Type> out_types;
      std::vector<std::vector<int>> out_shapes;

      auto dest = node->outlinks_in_order(true)[0]->sink()->safe_as<NodeData>();
      CHECK(dest);
      out_types.push_back(this->type_dict_.at(dest->id()));
      out_shapes.push_back(this->shape_dict_.at(dest->id()));

      auto impl =
          OpStrategy::SelectImpl(strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, target_));
      common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});

      // if node is master node for schedule
      if (group->master_nodes_.count(node)) {
        C = impl->fschedule(C);
      }

      // as elementwise only has one output
      CHECK_EQ(C.size(), 2);
      Expr out                  = C[0];
      auto out_tensor           = out.as_tensor_ref();
      poly::StageMap tmp_stages = C.back();

      tensor_map[dest->id()] = out_tensor;
      stages->InsertLazily(out.as_tensor_ref(), tmp_stages[out_tensor]);
    }
  };

  auto schedule = [this, &group, &stages, &tensor_map](const Group& sub_group) {
    for (int idx = sub_group->nodes_.size() - 1; idx >= 0; --idx) {
      auto node = sub_group->nodes_[idx];

      if (group->master_nodes_.count(node)) {
        continue;
      }

      // if node is internal node, use local buffer cache interal result and use SimpleComputeAt.
      if (group->internal_nodes_.count(node) || sub_group->internal_nodes_.count(node)) {
        stages[tensor_map[node->id()]]->SetBuffer("local");
        auto node_data = GetNodeData(node);
        auto shape     = this->shape_dict_.at(node_data->id());
        // find consumer node.
        auto consumers = GetConsumer(node);
        for (auto consumer : consumers) {
          if (tensor_map.count(consumer->id())) {
            stages[tensor_map[node->id()]]->SimpleComputeAt(stages[tensor_map[consumer->id()]], shape.size());
            break;
          }
        }
        continue;
      }

      // last node but not internal output, it's multi-output node
      if (idx == 0) {
        auto master_node      = *group->master_nodes_.begin();
        auto master_node_data = GetNodeData(master_node);
        auto shape            = this->shape_dict_.at(master_node_data->id());
        stages[tensor_map[node->id()]]->SimpleComputeAt(stages[tensor_map[master_node->id()]], shape.size());
        continue;
      }

      // others elemenwise node use compute-inline
      stages[tensor_map[node->id()]]->ComputeInline();
    }
  };

  if (group->fused_sub_groups_.size() == 0) {
    compute(group);
    schedule(group);
  } else {
    for (auto& sub_group : group->fused_sub_groups_) {
      compute(sub_group);
      schedule(sub_group);
    }
  }

  for (auto& var_id : fetch_var_ids_) {
    auto tensor = tensor_map[var_id];
    func_args.push_back(tensor);
    // recored tensor
    tensor_map[var_id] = tensor;
  }

  for (auto& node : group->output_nodes_) {
    if (fetch_var_ids_.count(node->id())) {
      continue;
    }
    auto tensor = tensor_map[node->id()];
    func_args.push_back(tensor);
    // recored tensor
    tensor_map[node->id()] = tensor;
  }

  return lang::Lower(func_name, stages, func_args, {}, {}, nullptr, this->target_);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
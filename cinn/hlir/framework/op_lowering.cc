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

using Comparator = Graph::Group::SharedGroupComparator;
using Hasher     = Graph::Group::SharedGroupHasher;

NodeData* GetNodeData(Node* node) {
  auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
  CHECK(node_data);
  return node_data;
}

std::vector<NodeData*> GetAllNodeData(Node* node) {
  std::vector<NodeData*> node_datas;
  for (auto& link : node->outlinks_in_order()) {
    auto node_data = link->sink()->safe_as<NodeData>();
    CHECK(node_data);
    node_datas.push_back(node_data);
  }

  return node_datas;
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

OpLowerer::OpLowerer(const absl::flat_hash_map<std::string, Type>& type_dict,
                     const absl::flat_hash_map<std::string, shape_t>& shape_dict,
                     const Target& target)
    : type_dict_(type_dict), shape_dict_(shape_dict), target_(target) {}

std::vector<ir::LoweredFunc> OpLowerer::Lower(GroupPtr& group) {
  VLOG(3) << "Lowering Group : " << group->group_id << " , Op Pattern : " << group->op_pattern_kind;
  switch (group->op_pattern_kind) {
    case framework::kElemWise:
    case framework::kBroadcast:
    case framework::kInjective:
      return LowerOp(&OpLowerer::ElementwiseCompute, &OpLowerer::ElementwiseSchedule, group);
    case framework::kCommReduce:
      return LowerOp(&OpLowerer::ReduceCompute, &OpLowerer::ReduceSchedule, group);
    case framework::kOutEWiseFusable:
      return LowerOp(&OpLowerer::OutEWiseFusableCompute, &OpLowerer::OutEWiseFusableSchedule, group);
    case framework::kOpaque:
      return LowerOpaqueOp(group);
    default:
      LOG(FATAL) << "Group Pattern Kind Is Unknown!";
  }
}

// fusion op lowering
std::vector<ir::LoweredFunc> OpLowerer::LowerOp(ComputeFunction compute, ScheduleFunction schedule, GroupPtr& group) {
  poly::StageMap stages;
  std::vector<ir::Tensor> func_args;
  std::unordered_map<std::string, ir::Tensor> tensor_map;

  // do compute.
  if (group->fused_sub_groups.size() == 0) {
    (this->*compute)(stages, func_args, tensor_map, group, group);
  } else {
    for (auto& sub_group : group->fused_sub_groups) {
      (this->*compute)(stages, func_args, tensor_map, group, sub_group);
    }
  }

  // do schedule.
  if (group->fused_sub_groups.size() == 0) {
    (this->*schedule)(stages, tensor_map, group, group);
  } else {
    for (auto& sub_group : group->fused_sub_groups) {
      (this->*schedule)(stages, tensor_map, group, sub_group);
    }
  }

  for (auto& args : func_args) {
    // input node data name.
    group->input_names.push_back(args->name);
  }

  for (auto& node : group->output_nodes) {
    // output node data name.
    for (auto node_data : GetAllNodeData(node)) {
      group->output_names.push_back(node_data->id());
    }
    // collect all output tensor.
    std::string post   = "";
    std::string prefix = GetNodeData(node)->id();
    for (int idx = 0;; ++idx) {
      if (!tensor_map.count(prefix + post)) {
        break;
      }
      auto tensor = tensor_map[prefix + post];
      // if tensor is with buffer, it's not a output.
      if (!tensor->buffer.defined() && !stages[tensor]->inlined()) {
        func_args.push_back(tensor);
      }
      // update post
      post = "_" + std::to_string(idx);
    }
  }

  return lang::LowerVec(group->GetFuncName(), stages, func_args, {}, {}, nullptr, this->target_);
}

std::vector<ir::Tensor> OpLowerer::CollectInputTensor(std::vector<ir::Tensor>& func_args,
                                                      std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                                      const Node* node) {
  std::vector<ir::Tensor> tensor_inputs;
  // get all input nodes
  for (auto& link : node->inlinks_in_order(true)) {
    auto source = link->source();
    CHECK(source);
    auto source_data = source->safe_as<NodeData>();
    CHECK(source_data);

    if (tensor_map.count(source_data->id())) {
      tensor_inputs.push_back(tensor_map[source_data->id()]);
    } else {
      auto dtype = this->type_dict_.at(source_data->id());
      CHECK(dtype == Float(32) || dtype.is_bool() || dtype == Int(32))
          << "The dtype of node " << source_data->id()
          << " is not float or bool or int! Other dtype is not implemented yet.";
      ir::Tensor tensor;
      if (dtype == Float(32)) {
        tensor = lang::Placeholder<float>(source_data->id(), this->shape_dict_.at(source_data->id()));
      } else if (dtype.is_bool()) {
        tensor = lang::Placeholder<bool>(source_data->id(), this->shape_dict_.at(source_data->id()));
      } else if (dtype == Int(32)) {
        tensor = lang::Placeholder<int>(source_data->id(), this->shape_dict_.at(source_data->id()));
      }
      tensor_map[source_data->id()] = tensor;
      tensor_inputs.push_back(tensor);
      // recored func input args
      func_args.push_back(tensor);
    }
  }

  return tensor_inputs;
}

void OpLowerer::ElementwiseCompute(poly::StageMap& stages,
                                   std::vector<ir::Tensor>& func_args,
                                   std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                   const GroupPtr& group,
                                   const GroupPtr& sub_group) {
  VLOG(3) << "ElementwiseCompute Group : " << sub_group->group_id;
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);
    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(func_args, tensor_map, node));
    for (auto& tensor : tensor_inputs) {
      stages->InsertLazily(tensor);
      cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    }

    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;

    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));

    auto impl =
        OpStrategy::SelectImpl(strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, this->target_));
    // do compute
    common::CINNValuePack value_pack = impl->fcompute(common::CINNValuePack{cinn_inputs});

    if (group->master_nodes.count(node)) {
      // do shedule
      value_pack = impl->fschedule(value_pack);
    }

    CHECK(value_pack.size() == 2);
    Expr out                  = value_pack[0];
    poly::StageMap tmp_stages = value_pack.back();

    tensor_map[node_data->id()] = out.as_tensor_ref();
    stages->InsertLazily(out.as_tensor_ref(), tmp_stages[out.as_tensor_ref()]);
  }
}

void OpLowerer::ElementwiseSchedule(poly::StageMap& stages,
                                    std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                    const GroupPtr& group,
                                    const GroupPtr& sub_group) {
  VLOG(3) << "ElementwiseSchedule Group : " << sub_group->group_id;
  auto master_node      = *group->master_nodes.begin();
  auto master_node_data = GetNodeData(master_node);
  auto master_stage     = stages[tensor_map[master_node_data->id()]];
  auto master_shape     = this->shape_dict_.at(master_node_data->id());
  for (auto& node : sub_group->nodes) {
    auto node_data  = GetNodeData(node);
    auto node_stage = stages[tensor_map[node_data->id()]];
    auto node_shape = this->shape_dict_.at(node_data->id());
    // if group master node
    if (group->master_nodes.count(node)) {
      continue;
    }

    if (master_shape != node_shape) {
      CHECK(!group->output_nodes.count(node)) << node->id() << " is to be broadcasted, it can't be output!";
      node_stage->ComputeInline();
      continue;
    }

    CHECK(master_shape == node_shape) << "node data shape must be equal to master node!";
    // if node is fringe node or internal node, fringe node is output node of sub-graph
    if (group->output_nodes.count(node) || group->internal_nodes.count(node) || sub_group->internal_nodes.count(node)) {
      // copy schedule from master node
      node_stage->CopyTransform(master_stage);
      node_stage->CopyLoopInfo(master_stage);
      // internal node use buffer
      if (!group->output_nodes.count(node)) {
        node_stage->SetBuffer("local");
      }
      // compute at master node
      node_stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
      continue;
    }

    // others elemenwise internal node use compute-inline
    node_stage->ComputeInline();
  }
}

void OpLowerer::ReduceCompute(poly::StageMap& stages,
                              std::vector<ir::Tensor>& func_args,
                              std::unordered_map<std::string, ir::Tensor>& tensor_map,
                              const GroupPtr& group,
                              const GroupPtr& sub_group) {
  VLOG(3) << "ReduceCompute Group : " << sub_group->group_id;
  auto& cinn_strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  Node* reducer = nullptr;
  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);

    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(func_args, tensor_map, node));
    for (auto& tensor : tensor_inputs) {
      stages->InsertLazily(tensor);
      cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    }

    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;

    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));

    auto impl =
        OpStrategy::SelectImpl(cinn_strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, target_));
    // do compute
    common::CINNValuePack value_pack = impl->fcompute(common::CINNValuePack{cinn_inputs});

    CHECK_GE(value_pack.size(), 2UL);
    CHECK_LE(value_pack.size(), 5UL);
    poly::StageMap tmp_stages = value_pack.back();

    std::string post = "";
    for (int idx = 0; idx < value_pack.size() - 1; ++idx) {
      Expr expr = value_pack[idx];
      stages->InsertLazily(expr.as_tensor_ref(), tmp_stages[expr.as_tensor_ref()]);
      tensor_map[node_data->id() + post] = expr.as_tensor_ref();
      // As op may has more than 1 output tensor, using id + "_0"/"_1" as key.
      post = "_" + std::to_string(idx);
    }
    value_pack.back() = CINNValue(stages);

    // node is kCommReduce
    if (op_pattern_dict[node->op()] == framework::kCommReduce) {
      reducer = node;
      // do schedule
      value_pack = impl->fschedule(value_pack);
    } else if (group->master_nodes.count(node)) {
      Expr out = value_pack[0];
      // node is master node, copy schedule from reduce node
      if (reducer) {
        auto reducer_data = GetNodeData(reducer);
        stages[out.as_tensor_ref()]->CopyTransform(stages[tensor_map[reducer_data->id()]]);
        stages[out.as_tensor_ref()]->CopyLoopInfo(stages[tensor_map[reducer_data->id()]]);
      } else {
        bool copied_transform = false;
        for (auto rnode : group->master_nodes) {
          if (op_pattern_dict[rnode->op()] == framework::kCommReduce) {
            auto rnode_data = GetNodeData(rnode);
            if (!tensor_map.count(rnode_data->id())) {
              continue;
            }
            stages[out.as_tensor_ref()]->CopyTransform(stages[tensor_map[rnode_data->id()]]);
            stages[out.as_tensor_ref()]->CopyLoopInfo(stages[tensor_map[rnode_data->id()]]);
            copied_transform = true;
            break;
          }
        }
        CHECK(copied_transform) << "master node fail to copy transfrom from reduce node!";
      }
    }
  }
}

void OpLowerer::ReduceSchedule(poly::StageMap& stages,
                               std::unordered_map<std::string, ir::Tensor>& tensor_map,
                               const GroupPtr& group,
                               const GroupPtr& sub_group) {
  VLOG(3) << "ReduceSchedule Group : " << sub_group->group_id;
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  // assign reduce input tensor schedule, do loop transform.
  auto OrderAssignReduce = [this, &stages](
                               poly::Stage* stage, const std::vector<int>& axes, const bool just_reorder = false) {
    // reorder none-last reduce axis to last.
    // like: shape = [16,16,16,16,16],axes = [1,3] -> new order = [0, 2, 4, 1, 3].
    std::vector<int> order;
    int n_out_dims = stage->n_out_dims();
    for (int idx = 0; idx < n_out_dims; ++idx) {
      if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
        order.push_back(idx);
      }
    }
    for (auto axis : axes) {
      order.push_back(axis);
    }
    stage->Reorder(order);

    if (just_reorder) {
      return;
    }

    // fuse others none-reduce axis.
    int last_dimension_num = n_out_dims - axes.back() - 1;
    int index              = n_out_dims - last_dimension_num - axes.size();

    // fuse last_dimension_num - 1 times
    for (auto idx = index; idx < index + last_dimension_num - 1; ++idx) {
      stage->Fuse(index, index + 1);
    }

    if (stage->GetDimRange(index) > this->target_.max_num_threads()) {
      stage->Split(index, this->target_.max_num_threads());
    }

    // fuse index - 1 times
    for (int idx = 0; idx < index - 1; ++idx) {
      stage->Fuse(0, 1);
    }
  };

  auto WithoutLastDimInReduce = [](const std::vector<int>& inshape, const std::vector<int>& axes) {
    // if last axis is in reduce.
    if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
        std::find(axes.begin(), axes.end(), -1) != axes.end()) {
      return false;
    }

    int sum_last_axes = 1;
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      sum_last_axes *= inshape[idx];
    }

    if (sum_last_axes > 1) {
      return true;
    } else {
      return false;
    }
  };

  auto ScheduleAssignReduceWithoutLast =
      [this, OrderAssignReduce](poly::Stage* stage, const std::vector<int>& inshape, const std::vector<int>& axes) {
        int lane            = 1;
        int max_num_threads = this->target_.max_num_threads();
        for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
          lane *= inshape[idx];
        }
        CHECK_LT(lane, max_num_threads / 2) << "Parallel threads must less than max_num_threads/2 on gpu!";
        int pos   = 0;
        int index = axes.size() - 1;
        for (; index >= 0; --index) {
          if (index + 1 < axes.size() && axes[index] != axes[index + 1] - 1) {
            pos = axes[index + 1];
            break;
          }

          lane *= inshape[axes[index]];
          if (lane > max_num_threads / 2) {
            pos = axes[index];
            break;
          }

          if (index == 0) {
            pos = axes[0];
          }
        }

        if (lane > max_num_threads / 2) {
          int prefix = inshape[axes[index]];
          int tail   = lane / prefix;
          for (int idx = max_num_threads / tail; idx > (max_num_threads / 2) / tail; --idx) {
            if (prefix % idx == 0) {
              stage->Split(axes[index], idx);
              break;
            }
            CHECK_GT(idx - 1, (max_num_threads / 2) / tail) << "idx should greater than (max_num_threads / 2) / tail.";
          }
        }

        // insert 1
        for (int idx = 0; idx < axes.size() - 1 - index; ++idx) {
          stage->Split(pos, stage->GetDimRange(pos));
        }

        OrderAssignReduce(stage, axes);
      };

  auto ScheduleAssignReduceWithLast =
      [this, OrderAssignReduce](poly::Stage* stage, const std::vector<int>& inshape, const std::vector<int>& axes) {
        // find first reduce and second reduce axis.
        int lane             = 1;
        int index            = static_cast<int>(axes.size()) - 1;
        auto max_num_threads = this->target_.max_num_threads();
        for (; index >= 0; --index) {
          if (index + 1 < axes.size() && axes[index] != axes[index + 1] - 1) {
            break;
          }
          lane *= inshape[axes[index]];
          if (index == 0 && lane <= max_num_threads) {
            LOG(FATAL) << "Error! lane is less equal than max_num_threads, Please check!";
          }
          if (lane >= max_num_threads / 2) {
            if (lane <= max_num_threads) {
              --index;
            }
            break;
          }
        }
        std::vector<int> first_axes(axes.begin(), axes.begin() + index + 1);
        if (lane > max_num_threads) {
          // last reduce axis size > 1024
          if (index == static_cast<int>(axes.size()) - 1) {
            int idx = max_num_threads;
            do {
              if (lane % idx == 0) {
                stage->Split(axes[index], idx);
                break;
              }
              --idx;
            } while (idx >= max_num_threads / 2);
            // if can't be divide by(1024, 512), it's shouldn't be fused.
            CHECK_GE(idx, max_num_threads / 2) << "Check bounds exist, can't fuse!";
          } else {
            int axis   = axes[index];
            int prefix = inshape[axis];
            int tail   = lane / prefix;
            for (int idx = max_num_threads / tail; idx > (max_num_threads / 2) / tail; --idx) {
              if (prefix % idx == 0) {
                stage->Split(axis, idx);
                break;
              }
              CHECK_GT(idx, (max_num_threads / 2) / tail) << "Error, it's shouldn't fuse!";
            }
          }
          OrderAssignReduce(stage, first_axes);
        } else {
          int fuse_times = axes.size() - (index + 1) - 1;
          for (int idx = 0; idx < fuse_times; ++idx) {
            stage->Fuse(axes[index + 1], axes[index + 1] + 1);
          }
          OrderAssignReduce(stage, first_axes, true);
        }
      };

  Node* master_node = nullptr;
  for (auto node : group->master_nodes) {
    if (op_pattern_dict[node->op()] != framework::kCommReduce) {
      master_node = node;
      break;
    }
  }

  // if not find master node, using last kCommReduce as master node.
  if (!master_node) {
    if (group->fused_sub_groups.empty()) {
      master_node = group->nodes.back();
    } else {
      master_node = group->fused_sub_groups.back()->nodes.back();
    }
    CHECK_EQ(op_pattern_dict[master_node->op()], framework::kCommReduce) << "Master Node Type Must Be Reduce!";
  }
  auto master_node_data = GetNodeData(master_node);
  auto master_stage     = stages[tensor_map[master_node_data->id()]];

  Node* master_reducer = op_pattern_dict[master_node->op()] == framework::kCommReduce ? master_node : nullptr;
  // find the reducer that link to master node.
  if (!master_reducer) {
    for (auto reducer : group->master_nodes) {
      if (op_pattern_dict[reducer->op()] == framework::kCommReduce) {
        master_reducer = reducer;
        break;
      }
    }
  }
  CHECK(master_reducer) << "Can't find Master reducer!";

  auto master_reducer_data  = GetNodeData(master_reducer);
  auto master_reducer_stage = stages[tensor_map[master_reducer_data->id()]];
  auto master_reducer_axes  = absl::get<std::vector<int>>(master_reducer->attrs.attr_store.at("dim"));
  auto master_reducer_shape = this->shape_dict_.at(master_reducer->inlinks_in_order()[0]->source()->id());
  // update sync thread depend.
  for (auto stage : stages) {
    if (stage.first.find("syncthreads") != std::string::npos) {
      stage.second->CtrlDepend(tensor_map[master_reducer_data->id() + "_0"]);
    }
  }

  VLOG(3) << "master node : " << master_node->id() << " ,reducer node : " << master_reducer->id();
  for (auto& node : sub_group->nodes) {
    VLOG(3) << "Schedule node -> " << node->id();
    auto node_data = GetNodeData(node);
    auto stage     = stages[tensor_map[node_data->id()]];
    // if node is kCommReduce
    if (node == master_node) {
      continue;
    }
    // for x86 schedule.
    if (this->target_ == common::DefaultHostTarget()) {
      if (op_pattern_dict[node->op()] == framework::kCommReduce) {
        if (!group->output_nodes.count(node)) {
          stage->SetBuffer("local");
        }
        if (node == master_reducer) {
          stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
        } else {
          stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - 1);
        }
        continue;
      }

      if (group->output_nodes.count(node) || group->internal_nodes.count(node) ||
          sub_group->internal_nodes.count(node)) {
        if (!group->output_nodes.count(node)) {
          stage->SetBuffer("local");
        }
        if (this->shape_dict_.at(node_data->id()) == this->shape_dict_.at(master_node_data->id())) {
          stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
        } else {
          if (stage->n_out_dims() == master_reducer_stage->n_out_dims() - 1) {
            stage->Split(0, stage->GetDimRange(0));
          }
          if (stage->n_out_dims() == master_reducer_stage->n_out_dims()) {
            std::vector<int> order;
            for (int idx = 0; idx < master_reducer_shape.size(); ++idx) {
              if (std::find(master_reducer_axes.begin(), master_reducer_axes.end(), idx) == master_reducer_axes.end()) {
                order.push_back(idx);
              }
            }
            for (auto axis : master_reducer_axes) {
              order.push_back(axis);
            }
            stage->Reorder(order);
            stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - 1);
          } else {
            stage->ComputeInline();
          }
        }
        continue;
      }

      stage->ComputeInline();
      continue;
    }

    // if node is kCommReduce
    if (op_pattern_dict[node->op()] == framework::kCommReduce) {
      VLOG(3) << "Reduce Schedule for Reduce Type!";
      // if node is not output node, set buffer.
      if (!group->output_nodes.count(node)) {
        stage->SetBuffer("local");
      }
      // last dimension is not in reduce.
      if (WithoutLastDimInReduce(master_reducer_shape, master_reducer_axes)) {
        // compute at last dimension
        if (node == master_reducer) {
          stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
        } else {
          // if don't use block shuffle reduce.
          if (!tensor_map.count(node_data->id() + "_1")) {
            if (master_reducer_stage->n_out_dims() > 1) {
              stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - 1);
            }
          } else {
            auto stage_1 = stages[tensor_map[node_data->id() + "_0"]];
            auto stage_2 = stages[tensor_map[master_reducer_data->id() + "_0"]];
            // compute at master reducer
            stage_1->SimpleComputeAt(stage_2, stage_2->n_out_dims() - 1);
            // delete stage_1 compute at stage
            stage_1->GetComputeAts().erase(stage->id());
            stage->CtrlDepend(tensor_map[master_reducer_data->id() + "_0"]);
            // comput at master stage
            stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - 1);
          }
        }
      } else {
        if (node == master_reducer) {
          stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
        } else if (tensor_map.count(node_data->id() + "_1")) {
          auto stage_1 = stages[tensor_map[node_data->id() + "_1"]];
          auto stage_2 = stages[tensor_map[master_reducer_data->id() + "_1"]];
          // compute at master reducer
          stage_1->SimpleComputeAt(stage_2, stage_2->n_out_dims() - 1);
          // delete stage_1 compute at stage_0
          auto stage_0 = stages[tensor_map[node_data->id() + "_0"]];
          stage_1->GetComputeAts().erase(stage_0->id());
          stage_0->CtrlDepend(tensor_map[master_reducer_data->id() + "_1"]);

          stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - 1);
        } else if (tensor_map.count(node_data->id() + "_0")) {
          stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - 1);
        } else {
          LOG(FATAL) << "Error! Unkown Reduce Type, Please Check!";
        }
      }
      continue;
    }

    // if node is internal node or output, try to copy schedule from fellow node
    if (group->output_nodes.count(node) || group->internal_nodes.count(node) || sub_group->internal_nodes.count(node)) {
      VLOG(3) << "Reduce Schedule for Elementwise Type";
      // if node is not output node, set buffer.
      if (!group->output_nodes.count(node)) {
        stage->SetBuffer("local");
      }
      // node is after reduce
      if (this->shape_dict_.at(node_data->id()) == this->shape_dict_.at(master_node_data->id())) {
        stage->CopyTransform(master_stage);
        stage->CopyLoopInfo(master_stage);
        // fringe node with no consumer
        stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
        continue;
      }
      // node is before reduce.
      if (WithoutLastDimInReduce(master_reducer_shape, master_reducer_axes)) {
        VLOG(3) << "Reduce Schedule for WithoutLastDimInReduce";
        // if used block shuffle reduce
        if (tensor_map.count(master_reducer_data->id() + "_1")) {
          ScheduleAssignReduceWithoutLast(stage, master_reducer_shape, master_reducer_axes);
          auto stage_0 = stages[tensor_map[master_reducer_data->id() + "_0"]];
          if (stage->n_out_dims() < stage_0->n_out_dims()) {
            stage->Split(0, stage->GetDimRange(0));
          }
          CHECK_EQ(stage->n_out_dims(), stage_0->n_out_dims()) << "stage and stage_0's n_out_dims must be equal!";
          stage->SimpleComputeAt(stage_0, stage_0->n_out_dims() - 1);
        } else {
          OrderAssignReduce(stage, master_reducer_axes);
          if (stage->n_out_dims() < master_reducer_stage->n_out_dims()) {
            stage->Split(0, stage->GetDimRange(0));
          }
          CHECK_EQ(stage->n_out_dims(), master_reducer_stage->n_out_dims())
              << "stage and master_reducer_stage's n_out_dims must be equal!";
          stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - 1);
        }
      } else {
        VLOG(3) << "Reduce Schedule for WithLastDimInReduce";
        if (tensor_map.count(master_reducer_data->id() + "_1")) {
          ScheduleAssignReduceWithLast(stage, master_reducer_shape, master_reducer_axes);
          auto reducer_stage = stages[tensor_map[master_reducer_data->id() + "_1"]];
          if (stage->n_out_dims() < reducer_stage->n_out_dims()) {
            stage->Split(0, stage->GetDimRange(0));
          }
          CHECK_EQ(stage->n_out_dims(), reducer_stage->n_out_dims())
              << "stage and reducer_stage's n_out_dims must be equal!";
          stage->SimpleComputeAt(reducer_stage, reducer_stage->n_out_dims() - 1);
        } else {
          // compute at reduce node
          auto reducer_stage = stages[tensor_map[master_reducer_data->id() + "_0"]];
          stage->CopyTransform(reducer_stage);
          stage->CopyLoopInfo(reducer_stage);
          stage->SimpleComputeAt(reducer_stage, reducer_stage->n_out_dims() - 1);
        }
      }
      continue;
    }
    // others elemenwise internal node use compute-inline
    stage->ComputeInline();
  }
}

void OpLowerer::OutEWiseFusableCompute(poly::StageMap& stages,
                                       std::vector<ir::Tensor>& func_args,
                                       std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                       const GroupPtr& group,
                                       const GroupPtr& sub_group) {
  VLOG(3) << "OutEWiseFusableCompute Group : " << sub_group->group_id;
  auto& cinn_strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);

    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(func_args, tensor_map, node));
    for (auto& tensor : tensor_inputs) {
      stages->InsertLazily(tensor);
      cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    }

    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;

    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));

    auto impl =
        OpStrategy::SelectImpl(cinn_strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, target_));
    // do compute
    common::CINNValuePack value_pack = impl->fcompute(common::CINNValuePack{cinn_inputs});

    CHECK_GE(value_pack.size(), 2);
    ir::Expr out              = value_pack[0];
    poly::StageMap tmp_stages = value_pack.back();
    // node is kCommReduce
    if (op_pattern_dict[node->op()] == framework::kOutEWiseFusable) {
      // do schedule
      value_pack = impl->fschedule(value_pack);
    } else if (group->master_nodes.count(node)) {
      // node is master node, copy schedule from OutEWiseFusable node
      for (auto fnode : group->master_nodes) {
        if (op_pattern_dict[fnode->op()] == framework::kOutEWiseFusable) {
          auto fnode_data = GetNodeData(fnode);
          tmp_stages[out.as_tensor_ref()]->CopyTransform(stages[tensor_map[fnode_data->id()]]);
          tmp_stages[out.as_tensor_ref()]->CopyLoopInfo(stages[tensor_map[fnode_data->id()]]);
          break;
        }
      }
    }

    std::string postfix = "";
    for (auto idx = 0; idx < value_pack.size() - 1; ++idx) {
      ir::Expr out                          = value_pack[idx];
      tensor_map[node_data->id() + postfix] = out.as_tensor_ref();
      stages->InsertLazily(out.as_tensor_ref(), tmp_stages[out.as_tensor_ref()]);
      // update postfix
      postfix = "_" + std::to_string(idx);
    }
  }
}

void OpLowerer::OutEWiseFusableSchedule(poly::StageMap& stages,
                                        std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                        const GroupPtr& group,
                                        const GroupPtr& sub_group) {
  VLOG(3) << "OutEWiseFusableSchedule Group : " << sub_group->group_id;
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  Node* master_node     = nullptr;
  for (auto node : group->master_nodes) {
    if (op_pattern_dict[node->op()] != framework::kOutEWiseFusable) {
      master_node = node;
      break;
    }
  }

  // if not find master node, using last kOutEWiseFusable as master node.
  if (!master_node) {
    if (group->fused_sub_groups.empty()) {
      master_node = group->nodes.back();
    } else {
      master_node = group->fused_sub_groups.back()->nodes.back();
    }
    CHECK_EQ(op_pattern_dict[master_node->op()], framework::kOutEWiseFusable)
        << "Master Node Type Must Be OutEWiseFusable!";
  }

  auto master_node_data = GetNodeData(master_node);
  auto master_stage     = stages[tensor_map[master_node_data->id()]];

  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);
    auto stage     = stages[tensor_map[node_data->id()]];
    // if node is master node.
    if (node == master_node) {
      continue;
    }

    // if node is kOutEWiseFusable
    if (op_pattern_dict[node->op()] == framework::kOutEWiseFusable) {
      // if node is not output nodes
      if (!group->output_nodes.count(node)) {
        tensor_map[node_data->id()]->WithBuffer("local");
      }
      // use compute at master node
      stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
      continue;
    }

    // if node is internal node or output, try to copy schedule from fellow node
    if (group->output_nodes.count(node) || group->internal_nodes.count(node) || sub_group->internal_nodes.count(node)) {
      // copy transform from master node
      stage->CopyTransform(master_stage);
      stage->CopyLoopInfo(master_stage);

      if (group->internal_nodes.count(node) || sub_group->internal_nodes.count(node)) {
        stage->SetBuffer("local");
      }
      // fringe node with no consumer
      stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
      continue;
    }
    // others elemenwise internal node use compute-inline
    stage->ComputeInline();
  }
}

std::vector<ir::LoweredFunc> OpLowerer::LowerOpaqueOp(GroupPtr& group) {
  VLOG(3) << "LowerOpaqueOp Group : " << group->group_id;
  // get input tensor and output tensor
  std::vector<ir::Tensor> func_args;
  CHECK_EQ(group->nodes.size(), 1) << "fusion op exist more than 1 op.";
  auto& cinn_strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  auto node = *group->nodes.begin();
  std::vector<ir::Tensor> tensor_inputs;
  std::vector<common::CINNValue> cinn_inputs;
  for (auto& link : node->inlinks_in_order(true)) {
    auto source = link->source();
    CHECK(source);
    auto source_data = source->safe_as<NodeData>();
    CHECK(source_data);

    auto tensor = lang::Placeholder<float>(source_data->id(), this->shape_dict_.at(source_data->id()));
    tensor_inputs.push_back(tensor);

    cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    // recored func input args
    func_args.push_back(tensor);
    // collect input node data name.
    group->input_names.push_back(tensor->name);
  }

  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;

  auto node_datas = GetAllNodeData(node);
  for (auto node_data : node_datas) {
    // collect output node data name.
    group->output_names.push_back(node_data->id());
    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));
  }

  auto impl =
      OpStrategy::SelectImpl(cinn_strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, target_));
  // do compute
  common::CINNValuePack value_pack = impl->fcompute(common::CINNValuePack{cinn_inputs});
  // do schedule
  value_pack = impl->fschedule(value_pack);

  CHECK(value_pack.size() >= 2);
  poly::StageMap stages = value_pack.back();
  // lazily insert input tensor.
  for (auto tensor_input : tensor_inputs) {
    stages->InsertLazily(tensor_input);
  }

  for (int idx = 0; idx < value_pack.size() - 1; ++idx) {
    Expr out    = value_pack[idx];
    auto tensor = out.as_tensor_ref();
    // collect output tensor
    if (!tensor->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) {
      func_args.push_back(tensor);
    }
  }

  return lang::LowerVec(group->GetFuncName(), stages, func_args, {}, {}, nullptr, this->target_);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

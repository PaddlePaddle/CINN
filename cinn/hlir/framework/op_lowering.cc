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

#include "cinn/optim/transform_gpu_forloop.h"

DECLARE_bool(cinn_ir_schedule);

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

NodeData* GetNodeData(const Node* node) {
  auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
  CHECK(node_data);
  return node_data;
}

std::vector<NodeData*> GetAllNodeData(const Node* node) {
  std::vector<NodeData*> node_datas;
  for (auto& link : node->outlinks_in_order(true)) {
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
  if (FLAGS_cinn_ir_schedule) {
    switch (group->op_pattern_kind) {
      case framework::kElemWise:
      case framework::kBroadcast:
      case framework::kInjective:
        return IRLowerOp(&OpLowerer::IRElementwiseCompute, &OpLowerer::IRElementwiseSchedule, group);
      case framework::kCommReduce:
        return IRLowerOp(&OpLowerer::IRReduceCompute, &OpLowerer::IRReduceSchedule, group);
      case framework::kOutEWiseFusable:
        LOG(FATAL) << "Group Pattern Kind kOutEWiseFusable Is Not Implemented!";
      case framework::kOpaque:
        return IRLowerOpaqueOp(group);
      default:
        LOG(FATAL) << "Group Pattern Kind Is Unknown!";
    }
  } else {
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
}

std::vector<ir::LoweredFunc> OpLowerer::IRLowerOp(IRComputeFunction compute,
                                                  IRScheduleFunction schedule,
                                                  GroupPtr& group) {
  poly::StageMap stages;
  std::vector<ir::Tensor> arg_tensors;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  // do compute.
  VLOG(3) << "group->fused_sub_groups.size() is : " << group->fused_sub_groups.size();
  std::vector<Expr> ast_exprs;
  if (group->fused_sub_groups.size() == 0) {
    ast_exprs = (this->*compute)(stages, arg_tensors, tensor_map, group, group);
  } else {
    for (auto& sub_group : group->fused_sub_groups) {
      auto exprs = (this->*compute)(stages, arg_tensors, tensor_map, group, sub_group);
      ast_exprs.insert(ast_exprs.end(), exprs.begin(), exprs.end());
    }
  }
  ir::ModuleExpr mod_expr(ast_exprs);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();

  Node* first  = nullptr;
  Node* second = nullptr;
  // do schedule.
  VLOG(3) << "Before IRLowerOp schedule";
  if (group->fused_sub_groups.size() == 0) {
    (this->*schedule)(ir_sch, tensor_map, group, group, first, second);
  } else {
    // do schedule from back to front.
    for (int idx = group->fused_sub_groups.size() - 1; idx >= 0; --idx) {
      (this->*schedule)(ir_sch, tensor_map, group, group->fused_sub_groups[idx], first, second);
    }
  }
  VLOG(3) << "After IRLowerOp schedule";
  // function args
  std::vector<ir::Argument> func_args;
  for (auto& args : arg_tensors) {
    // input node data name.
    group->input_names.push_back(args->name);
    // input args
    func_args.emplace_back(args->buffer, ir::Argument::IO::kInput);
  }

  for (auto& node : group->output_nodes) {
    // output node data name.
    for (auto node_data : GetAllNodeData(node)) {
      group->output_names.push_back(node_data->id());
    }
    // collect all output tensor.
    std::string post   = "";
    std::string prefix = GetNodeData(node)->id();
    for (int idx = 0; idx < 1; ++idx) {
      CHECK(tensor_map.count(prefix)) << "Can't find output tensor " << prefix;
      if (!tensor_map.count(prefix + post)) {
        break;
      }
      auto tensor = tensor_map[prefix + post];
      arg_tensors.push_back(tensor);
      // output args
      func_args.emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
      // update post
      post = "_" + std::to_string(idx);
    }
  }

  auto func_body    = ir_sch.GetModule().GetExprs().at(0);
  auto temp_buffers = lang::GetTempBuffers(arg_tensors, stages, func_body);
  auto func =
      ir::_LoweredFunc_::Make(group->GetFuncName(), func_args, ir_sch.GetModule().GetExprs().at(0), temp_buffers);
  func->PrepareBufferCastExprs();
#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&(func->body));
#endif
  func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();
  return {func};
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
      if (idx == 0) {
        CHECK(tensor_map.count(prefix)) << "Can't find output tensor " << prefix;
      }
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
    if (FLAGS_cinn_ir_schedule) {
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
      if (!tensor_map.count(source_data->id())) {
        tensor_map[source_data->id()] = tensor;
        // record func input args
        func_args.push_back(tensor);
      }
      tensor_inputs.push_back(tensor);
    } else {
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
        // record func input args
        func_args.push_back(tensor);
      }
    }
  }
  return tensor_inputs;
}

std::vector<Expr> OpLowerer::IRElementwiseCompute(poly::StageMap& stages,
                                                  std::vector<ir::Tensor>& func_tensors,
                                                  std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                                  const GroupPtr& group,
                                                  const GroupPtr& sub_group) {
  VLOG(3) << "ElementwiseCompute Group : " << sub_group->group_id;
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  std::vector<Expr> ast_exprs;
  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);
    CHECK_EQ(GetAllNodeData(node).size(), 1U);
    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(func_tensors, tensor_map, node));
    for (auto& tensor : tensor_inputs) {
      cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    }
    // set tensor name = node data name
    cinn_inputs.push_back(common::CINNValue(node_data->id()));

    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;
    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));
    auto impl =
        OpStrategy::SelectImpl(strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, this->target_));
    // do compute
    common::CINNValuePack pack = impl->fcompute(common::CINNValuePack{cinn_inputs});
    CHECK_EQ(pack.size(), 2U);

    Expr expr                  = pack[0];
    poly::StageMap node_stages = pack.back();
    tensor_inputs.push_back(expr.as_tensor_ref());
    tensor_map[node_data->id()] = expr.as_tensor_ref();

    auto func = lang::LowerVec("fn_" + node->id(), node_stages, tensor_inputs, {}, {}, nullptr, this->target_, true);
    CHECK_EQ(func.size(), 1);

    std::vector<common::CINNValue> schedule_inputs;
    // collect tensor
    for (int idx = 0; idx < pack.size() - 1; ++idx) {
      CHECK(pack[idx].is_tensor());
      schedule_inputs.push_back(common::CINNValue(pack[idx]));
    }
    for (auto& f : func) {
      schedule_inputs.push_back(common::CINNValue(f->body));
    }
    // do ast tree schedule
    common::CINNValuePack expr_pack = impl->fschedule(common::CINNValuePack{schedule_inputs});

    CHECK_EQ(expr_pack.size(), 1);
    Expr ast_expr = expr_pack[0];
    ast_exprs.push_back(ast_expr);
  }

  return ast_exprs;
}

void OpLowerer::IRElementwiseSchedule(ir::IRSchedule& ir_sch,
                                      std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                      const GroupPtr& group,
                                      const GroupPtr& sub_group,
                                      Node*&,
                                      Node*&) {
  VLOG(3) << "IRElementwiseSchedule Group : " << sub_group->group_id;
  auto master_node    = *group->master_nodes.begin();
  auto manster_tensor = tensor_map[GetNodeData(master_node)->id()];

  for (int idx = sub_group->nodes.size() - 1; idx >= 0; --idx) {
    auto node        = sub_group->nodes[idx];
    auto node_tensor = tensor_map[GetNodeData(node)->id()];
    if (group->master_nodes.count(node)) {
      continue;
    }

    // if node is fringe node or internal node, fringe node is output node of sub-graph
    if (group->output_nodes.count(node) || group->internal_nodes.count(node) || sub_group->internal_nodes.count(node)) {
      // internal node use buffer
      if (!group->output_nodes.count(node)) {
        auto node_block = ir_sch.GetBlock(node_tensor->name);
        ir_sch.SetBuffer(node_block, "local", true);
      }

      auto node_block   = ir_sch.GetBlock(node_tensor->name);
      auto master_loops = ir_sch.GetLoops(manster_tensor->name);
      ir_sch.SimpleComputeAt(node_block, master_loops.back());
      continue;
    }

    // others elemenwise internal node use compute-inline
    ir_sch.ComputeInline(ir_sch.GetBlock(node_tensor->name));
  }
}

std::vector<Expr> OpLowerer::IRReduceCompute(poly::StageMap& stages,
                                             std::vector<ir::Tensor>& func_args,
                                             std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                             const GroupPtr& group,
                                             const GroupPtr& sub_group) {
  VLOG(2) << "ReduceCompute Group : " << sub_group->group_id;
  auto& cinn_strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  std::vector<Expr> ast_exprs;
  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);
    VLOG(3) << node->id();

    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(func_args, tensor_map, node));
    for (auto& tensor : tensor_inputs) {
      cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    }
    cinn_inputs.push_back(common::CINNValue(node_data->id()));

    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;

    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));

    auto impl =
        OpStrategy::SelectImpl(cinn_strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, target_));
    // do compute
    common::CINNValuePack pack = impl->fcompute(common::CINNValuePack{cinn_inputs});

    CHECK_GE(pack.size(), 2UL);
    CHECK_LE(pack.size(), 5UL);
    poly::StageMap tmp_stages = pack.back();

    std::string post = "";
    for (int idx = 0; idx < pack.size() - 1; ++idx) {
      Expr expr                          = pack[idx];
      tensor_map[node_data->id() + post] = expr.as_tensor_ref();
      // As op may has more than 1 output tensor, using id + "_0"/"_1" as key.
      post = "_" + std::to_string(idx);

      // Insert outout tensors
      if (!expr.as_tensor_ref()->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) {
        tensor_inputs.push_back(expr.as_tensor_ref());
      }
    }
    auto func = lang::LowerVec("fn_" + node->id(), tmp_stages, tensor_inputs, {}, {}, nullptr, this->target_, true);

    // node is kCommReduce
    if (op_pattern_dict[node->op()] == framework::kCommReduce) {
      std::vector<common::CINNValue> schedule_inputs;
      // collect tensor
      for (int idx = 0; idx < pack.size() - 1; ++idx) {
        CHECK(pack[idx].is_tensor());
        schedule_inputs.push_back(common::CINNValue(pack[idx]));
      }
      for (auto& f : func) {
        schedule_inputs.push_back(common::CINNValue(f->body));
      }
      // do ast tree schedule
      common::CINNValuePack expr_pack = impl->fschedule(common::CINNValuePack{schedule_inputs});
      // ast tree after schedule.
      Expr ast_expr = expr_pack[0];
      ast_exprs.push_back(ast_expr);
    } else if (group->master_nodes.count(node)) {
      // as master node should copy transform from reducer, left it to reduce schedule.
      ast_exprs.push_back(func[0]->body);
    } else {
      ast_exprs.push_back(func[0]->body);
    }
  }

  return ast_exprs;
}

void OpLowerer::IRReduceSchedule(ir::IRSchedule& ir_sch,
                                 std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                 const GroupPtr& group,
                                 const GroupPtr& sub_group,
                                 Node*& master,
                                 Node*& reducer) {
  auto& op_pattern_dict  = Operator::GetAttrs<OpPatternKind>("OpPattern");
  auto OrderAssignReduce = [this](ir::IRSchedule& ir_sch,
                                  const std::string& block_name,
                                  const std::vector<int>& axes,
                                  const bool just_reorder = false) {
    // reorder none-last reduce axis to last.
    // like: shape = [16,16,16,16,16],axes = [1,3] -> new order = [0, 2, 4, 1, 3].
    std::vector<int> order;
    int n_out_dims = ir_sch.GetLoops(block_name).size();
    for (int idx = 0; idx < n_out_dims; ++idx) {
      if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
        order.push_back(idx);
      }
    }
    for (auto axis : axes) {
      order.push_back(axis);
    }
    ir_sch.Reorder(ir_sch.GetBlock(block_name), order);

    if (just_reorder) {
      return;
    }
    // fuse others none-reduce axis.
    int last_dimension_num = n_out_dims - axes.back() - 1;
    int index              = n_out_dims - last_dimension_num - axes.size();

    // fuse last_dimension_num - 1 times
    for (auto idx = index; idx < index + last_dimension_num - 1; ++idx) {
      ir_sch.Fuse(block_name, {index, index + 1});
    }

    auto loops = ir_sch.GetLoops(block_name);

    if (ir::GetLoopExtent(loops[index]) > this->target_.max_num_threads()) {
      ir_sch.Split(block_name, index, {-1, this->target_.max_num_threads()});
    }

    // fuse index - 1 times
    for (int idx = 0; idx < index - 1; ++idx) {
      ir_sch.Fuse(block_name, {0, 1});
    }
  };

  auto WithoutLastDimInReduce = [](const std::vector<int>& inshape, std::vector<int>& axes) {
    // if last axis is in reduce.
    axes = axes.empty() ? inshape : axes;
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

  auto ScheduleAssignReduceWithoutLast = [this, OrderAssignReduce](ir::IRSchedule& ir_sch,
                                                                   const std::string& block_name,
                                                                   const std::vector<int>& inshape,
                                                                   std::vector<int>& axes) {
    axes                = axes.empty() ? inshape : axes;
    int lane            = 1;
    int max_num_threads = this->target_.max_num_threads();
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      lane *= inshape[idx];
    }
    CHECK_LE(lane, max_num_threads / 2) << "Parallel threads must less equal max_num_threads/2 on gpu!";
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
          ir_sch.Split(block_name, axes[index], {-1, idx});
          break;
        }
        CHECK_GT(idx - 1, (max_num_threads / 2) / tail) << "idx should greater than (max_num_threads / 2) / tail.";
      }
    }

    // insert 1
    for (int idx = 0; idx < axes.size() - 1 - index; ++idx) {
      auto loops = ir_sch.GetLoops(block_name);
      ir_sch.Split(block_name, pos, {-1, ir::GetLoopExtent(loops[pos])});
    }
    OrderAssignReduce(ir_sch, block_name, axes);
    // return insert 1
    int start_index = ir_sch.GetLoops(block_name).size() - axes.size();
    for (int idx = 0; idx < axes.size(); ++idx) {
      auto loops = ir_sch.GetLoops(block_name);
      if (ir::GetLoopExtent(loops[start_index]) == 1) {
        ir_sch.Fuse({loops[start_index - 1], loops[start_index]});
      } else {
        ++start_index;
      }
    }
  };

  auto ScheduleAssignReduceWithLast = [this, OrderAssignReduce](ir::IRSchedule& ir_sch,
                                                                const std::string& block_name,
                                                                const std::vector<int>& inshape,
                                                                std::vector<int>& axes) {
    // find first reduce and second reduce axis.
    axes                 = axes.empty() ? inshape : axes;
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
            ir_sch.Split(block_name, axes[index], {-1, idx});
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
            ir_sch.Split(block_name, axis, {-1, idx});
            break;
          }
          CHECK_GT(idx, (max_num_threads / 2) / tail) << "Error, it's shouldn't fuse!";
        }
      }
      OrderAssignReduce(ir_sch, block_name, first_axes);
    } else {
      int fuse_times = axes.size() - (index + 1) - 1;
      for (int idx = 0; idx < fuse_times; ++idx) {
        ir_sch.Fuse(block_name, {axes[index + 1], axes[index + 1] + 1});
      }
      OrderAssignReduce(ir_sch, block_name, first_axes, true);
    }
  };

  auto insert_for_node = [](ir::IRSchedule& ir_sch, std::string& block_name) {
    auto block      = ir_sch.GetBlock(block_name);
    auto root_block = ir_sch.GetRootBlock(block);

    CHECK(block->as<ir::ScheduleBlockRealize>());
    CHECK(block->as<ir::ScheduleBlockRealize>()->schedule_block->as<ir::ScheduleBlock>());

    // create var
    auto var = ir::Var(ir::Expr(0), ir::Expr(1), "i_0");
    block->as<ir::ScheduleBlockRealize>()->iter_values.push_back(var);
    block->as<ir::ScheduleBlockRealize>()->schedule_block->as<ir::ScheduleBlock>()->iter_vars.push_back(var);

    CHECK(block->as<ir::ScheduleBlockRealize>());
    CHECK(block->as<ir::ScheduleBlockRealize>()->schedule_block->as<ir::ScheduleBlock>());
    // create for and block node
    auto for_node =
        ir::For::Make(var, Expr(0), Expr(1), ir::ForType::Serial, ir::DeviceAPI::UNK, ir::Block::Make({block}));

    auto stmts = root_block->as<ir::ScheduleBlockRealize>()
                     ->schedule_block->as<ir::ScheduleBlock>()
                     ->body->as<ir::Block>()
                     ->stmts;
    std::vector<ir::Expr> n_stmts;
    for (auto expr : stmts) {
      if (!expr.As<ir::ScheduleBlockRealize>()) {
        n_stmts.push_back(expr);
        continue;
      }
      CHECK(expr.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>());
      if (expr.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name == block_name) {
        n_stmts.push_back(for_node);
        continue;
      }
      n_stmts.push_back(expr);
    }
    root_block->as<ir::ScheduleBlockRealize>()->schedule_block->as<ir::ScheduleBlock>()->body->as<ir::Block>()->stmts =
        n_stmts;
  };

  if (master == nullptr && reducer == nullptr) {
    auto blocks = ir_sch.GetAllBlocks();
    for (int idx = blocks.size() - 1; idx >= 0; --idx) {
      auto block = blocks[idx];
      CHECK(block->as<ir::ScheduleBlockRealize>());
      CHECK(block->as<ir::ScheduleBlockRealize>()->schedule_block->as<ir::ScheduleBlock>());
      if (!tensor_map.count(block->as<ir::ScheduleBlockRealize>()->schedule_block->as<ir::ScheduleBlock>()->name)) {
        continue;
      }
      for (auto node : group->master_nodes) {
        if (GetNodeData(node)->id() ==
            block->as<ir::ScheduleBlockRealize>()->schedule_block->as<ir::ScheduleBlock>()->name) {
          if (op_pattern_dict[node->op()] != framework::kCommReduce) {
            master = node;
            break;
          }

          if (op_pattern_dict[node->op()] == framework::kCommReduce) {
            reducer = node;
            break;
          }
        }
      }

      if (master && reducer) {
        break;
      }
    }
    // find master node.
    for (auto node : group->master_nodes) {
      if (op_pattern_dict[node->op()] != framework::kCommReduce) {
        master = node;
        break;
      }
    }
    // if not find, use reducer as master.
    if (!master) {
      if (group->fused_sub_groups.empty()) {
        master = group->nodes.back();
      } else {
        master = group->fused_sub_groups.back()->nodes.back();
      }
      CHECK_EQ(op_pattern_dict[master->op()], framework::kCommReduce) << "Master Node Type Must Be Reduce!";
    }

    // find master reducer node.
    reducer = op_pattern_dict[master->op()] == framework::kCommReduce ? master : nullptr;
    if (!group->fused_sub_groups.size()) {
      for (auto node : group->master_nodes) {
        if (op_pattern_dict[node->op()] == framework::kCommReduce) {
          reducer = node;
          break;
        }
      }
    }
    for (int idx = group->fused_sub_groups.size() - 1; idx >= 0 && !reducer; --idx) {
      if (group->fused_sub_groups[idx]->op_pattern_kind != framework::kCommReduce) {
        continue;
      }
      for (auto node : group->fused_sub_groups[idx]->master_nodes) {
        if (op_pattern_dict[node->op()] == framework::kCommReduce) {
          reducer = node;
          break;
        }
      }
    }
    CHECK(reducer) << "Can't find Master reducer!";

    if (reducer != master) {
      auto master_tensor = tensor_map[GetNodeData(master)->id()];
      // if master tensor shape = [1], insert for in ast tree
      if (master_tensor->shape[0] == ir::Expr(1)) {
        insert_for_node(ir_sch, master_tensor->name);
      }
      {
        auto master_block   = ir_sch.GetBlock(master_tensor->name);
        auto reducer_tensor = tensor_map[GetNodeData(reducer)->id()];
        auto reducer_block  = ir_sch.GetBlock(reducer_tensor->name);
        ir_sch.CopyTransformAndLoopInfo(master_block, reducer_block);
      }
    }
    // do reducer schedule.
    {
      auto reducer_data   = GetNodeData(reducer);
      auto reducer_tensor = tensor_map[reducer_data->id()];
      CHECK(reducer->attrs.attr_store.count("dim"));
      auto reducer_axes = absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
      CHECK(reducer->inlinks_in_order().size());
      CHECK(this->shape_dict_.count(reducer->inlinks_in_order()[0]->source()->id()));
      auto reducer_shape = this->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());

      if (reducer_axes.empty()) {
        for (int i = 0; i < reducer_shape.size(); ++i) {
          reducer_axes.emplace_back(i);
        }
      }

      bool without_last_dim = WithoutLastDimInReduce(reducer_shape, reducer_axes);

      std::unordered_set<Node*> visited_nodes;
      for (auto node : group->master_nodes) {
        VLOG(2) << "Schedule reduce node -> " << node->id();
        if (op_pattern_dict[node->op()] != framework::kCommReduce) {
          continue;
        }
        auto node_data   = GetNodeData(node);
        auto node_tensor = tensor_map[node_data->id()];

        if (!group->output_nodes.count(node)) {
          auto node_block = ir_sch.GetBlock(node_tensor->name);
          ir_sch.SetBuffer(node_block, "local", true);
        }
        if (node == reducer) {
          continue;
        }
        auto node_shape = this->shape_dict_.at(node->inlinks_in_order()[0]->source()->id());
        if (without_last_dim) {
          VLOG(2) << "Reduce Schedule WithoutLastDimInReduce";
          // find a shape to do simple compute at.
          auto tmp_reducer       = reducer;
          auto tmp_reducer_shape = reducer_shape;
          if (node_shape != reducer_shape) {
            // try to find the same shape reduce from visited_nodes
            for (auto visited : visited_nodes) {
              auto shape = this->shape_dict_.at(visited->inlinks_in_order()[0]->source()->id());
              if (shape == node_shape) {
                tmp_reducer       = visited;
                tmp_reducer_shape = shape;
                break;
              }
            }
          }
          visited_nodes.insert(node);
          auto tmp_reducer_data   = GetNodeData(tmp_reducer);
          auto tmp_reducer_tensor = tensor_map[tmp_reducer_data->id()];

          // using block shuffle reduce.
          if (tensor_map.count(reducer_data->id() + "_1")) {
            auto node_0_tensor = tensor_map[node_data->id() + "_0"];
            auto node_0_block  = ir_sch.GetBlock(node_0_tensor->name);

            auto tmp_reducer_0_tensor = tensor_map[tmp_reducer_data->id() + "_0"];
            auto tmp_reducer_0_loops  = ir_sch.GetLoops(tmp_reducer_0_tensor->name);

            if (tmp_reducer_shape == node_shape) {
              ir_sch.SimpleComputeAt(node_0_block, tmp_reducer_0_loops.back());
              // init compute at reduce
              int loop_depth = ir_sch.GetLoops(node_0_tensor->name + "__reduce_init").size();
              ir_sch.SimpleComputeAt(ir_sch.GetBlock(node_0_tensor->name + "__reduce_init"),
                                     ir_sch.GetLoops(node_0_tensor->name)[loop_depth - 1]);
            } else {
              if (tmp_reducer_0_tensor->shape.back() == node_0_tensor->shape.back()) {
                int num_reduce_axis = tmp_reducer_0_tensor->reduce_axis.size();
                CHECK_GE(static_cast<int>(tmp_reducer_0_loops.size()) - num_reduce_axis - 1, 0);
                ir_sch.SimpleComputeAt(node_0_block,
                                       tmp_reducer_0_loops[tmp_reducer_0_loops.size() - num_reduce_axis - 1]);
                // init compute at reduce
                int loop_depth = ir_sch.GetLoops(node_0_tensor->name + "__reduce_init").size();
                ir_sch.SimpleComputeAt(ir_sch.GetBlock(node_0_tensor->name + "__reduce_init"),
                                       ir_sch.GetLoops(node_0_tensor->name)[loop_depth - 1]);
              } else {
                CHECK_GE(static_cast<int>(tmp_reducer_0_loops.size()), 2);
                ir_sch.SimpleComputeAt(node_0_block, tmp_reducer_0_loops[0]);
              }
            }
            ir_sch.SimpleComputeAt(ir_sch.GetBlock(node_tensor->name),
                                   ir_sch.GetLoops(tmp_reducer_tensor->name).back());
          } else {
            if (tmp_reducer_shape == node_shape) {
              ir_sch.SimpleComputeAt(ir_sch.GetBlock(node_tensor->name),
                                     ir_sch.GetLoops(tmp_reducer_tensor->name).back());
            } else {
              int num_reduce_axis    = tmp_reducer_tensor->reduce_axis.size();
              auto tmp_reducer_loops = ir_sch.GetLoops(tmp_reducer_tensor->name);
              CHECK_GE(static_cast<int>(tmp_reducer_loops.size()) - num_reduce_axis - 1, 0);
              ir_sch.SimpleComputeAt(ir_sch.GetBlock(node_tensor->name),
                                     tmp_reducer_loops[tmp_reducer_loops.size() - num_reduce_axis - 1]);
            }
            // init compute at reduce
            int loop_depth = ir_sch.GetLoops(node_tensor->name + "__reduce_init").size();
            ir_sch.SimpleComputeAt(ir_sch.GetBlock(node_tensor->name + "__reduce_init"),
                                   ir_sch.GetLoops(node_tensor->name)[loop_depth - 1]);
          }
        } else {
          VLOG(2) << "Reduce Schedule WithLastDimInReduce";
          // if with column reduce behind.
          if (tensor_map.count(node_data->id() + "_1")) {
            auto reducer_1_tensor = tensor_map[reducer_data->id() + "_1"];
            auto reducer_0_tensor = tensor_map[reducer_data->id() + "_0"];

            auto node_1_tensor = tensor_map[node_data->id() + "_1"];
            auto node_0_tensor = tensor_map[node_data->id() + "_0"];

            auto node_block_1 = ir_sch.GetBlock(node_1_tensor->name);
            auto node_block_0 = ir_sch.GetBlock(node_0_tensor->name);
            auto node_block   = ir_sch.GetBlock(node_tensor->name);

            ir_sch.SimpleComputeAt(node_block, ir_sch.GetLoops(reducer_tensor->name).back());
            ir_sch.SimpleComputeAt(node_block_0, ir_sch.GetLoops(reducer_0_tensor->name).back());
            ir_sch.SimpleComputeAt(node_block_1, ir_sch.GetLoops(reducer_1_tensor->name).back());
            // init compute at reduce
            int loop_depth = ir_sch.GetLoops(node_1_tensor->name + "__reduce_init").size();
            ir_sch.SimpleComputeAt(ir_sch.GetBlock(node_1_tensor->name + "__reduce_init"),
                                   ir_sch.GetLoops(node_1_tensor->name)[loop_depth - 1]);
          } else if (tensor_map.count(node_data->id() + "_0")) {
            auto reducer_0_tensor = tensor_map[reducer_data->id() + "_0"];
            auto node_0_tensor    = tensor_map[node_data->id() + "_0"];

            auto node_0_block = ir_sch.GetBlock(node_0_tensor->name);
            auto node_block   = ir_sch.GetBlock(node_tensor->name);
            ir_sch.SimpleComputeAt(node_block, ir_sch.GetLoops(reducer_tensor->name).back());
            ir_sch.SimpleComputeAt(node_0_block, ir_sch.GetLoops(reducer_0_tensor->name).back());
          } else {
            LOG(FATAL) << "Error! Unkown Reduce Type, Please Check!";
          }
        }
      }

      if (without_last_dim) {
        if (tensor_map.count(reducer_data->id() + "_1")) {
          auto reducer_tensor = tensor_map[GetNodeData(reducer)->id()];
          auto reducer_loops  = ir_sch.GetLoops(reducer_tensor->name);
          ir_sch.SyncThreads(reducer_loops[0], false);
        }
      }
    }
  }

  auto master_data = GetNodeData(master);
  CHECK(master_data);
  CHECK(tensor_map.count(master_data->id()));
  auto master_tensor = tensor_map[master_data->id()];
  auto reducer_data  = GetNodeData(reducer);
  CHECK(reducer_data);
  CHECK(reducer->attrs.attr_store.count("dim"));
  auto reducer_axes = absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
  CHECK(reducer->inlinks_in_order().size());
  CHECK(this->shape_dict_.count(reducer->inlinks_in_order()[0]->source()->id()));
  auto reducer_shape = this->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());

  if (reducer_axes.empty()) {
    for (int i = 0; i < reducer_shape.size(); ++i) {
      reducer_axes.emplace_back(i);
    }
  }

  VLOG(2) << "master node : " << master->id() << " ,reducer node : " << reducer->id();
  for (int idx = sub_group->nodes.size() - 1; idx >= 0; --idx) {
    auto node = sub_group->nodes[idx];
    VLOG(2) << "Schedule node -> " << node->id();
    if (node == master) {
      continue;
    }
    if (op_pattern_dict[node->op()] == framework::kCommReduce) {
      continue;
    }

    auto node_data   = GetNodeData(node);
    auto node_tensor = tensor_map[node_data->id()];
    // for x86 schedule.
    if (this->target_ == common::DefaultHostTarget()) {
      LOG(FATAL) << "X86 Not implemented";
    }
    // if node is internal node or output, try to copy schedule from fellow node
    if (group->output_nodes.count(node) || group->internal_nodes.count(node) || sub_group->internal_nodes.count(node)) {
      VLOG(2) << "Reduce Schedule for Elementwise Type";
      // if node is not output node, set buffer.
      if (!group->output_nodes.count(node)) {
        auto node_block = ir_sch.GetBlock(node_tensor->name);
        ir_sch.SetBuffer(node_block, "local", true);
      }
      // node is after reduce
      if (this->shape_dict_.at(node_data->id()) == this->shape_dict_.at(master_data->id())) {
        if (node_tensor->shape[0] == ir::Expr(1)) {
          insert_for_node(ir_sch, node_tensor->name);
        }
        ir_sch.CopyTransformAndLoopInfo(node_tensor->name, master_tensor->name);
        auto node_block   = ir_sch.GetBlock(node_tensor->name);
        auto master_loops = ir_sch.GetLoops(master_tensor->name);
        ir_sch.SimpleComputeAt(node_block, master_loops.back());
        continue;
      }
      // node is before reduce.
      if (WithoutLastDimInReduce(reducer_shape, reducer_axes)) {
        VLOG(2) << "Reduce Schedule for WithoutLastDimInReduce";
        // find a shape to do simple compute at.
        auto tmp_reducer       = reducer;
        auto tmp_reducer_shape = reducer_shape;
        // node shape.
        auto node_shape = this->shape_dict_.at(node_data->id());
        if (node_shape != tmp_reducer_shape) {
          // try to find the same shape reduce from visited_nodes
          for (auto rnode : group->master_nodes) {
            if (op_pattern_dict[rnode->op()] != framework::kCommReduce) {
              continue;
            }
            auto shape = this->shape_dict_.at(rnode->inlinks_in_order()[0]->source()->id());
            if (shape == node_shape) {
              tmp_reducer       = rnode;
              tmp_reducer_shape = shape;
              break;
            }
          }
        }
        CHECK(node_shape == tmp_reducer_shape);
        auto tmp_reducer_data   = GetNodeData(tmp_reducer);
        auto tmp_reducer_tensor = tensor_map[tmp_reducer_data->id()];
        // if used block shuffle reduce
        if (tensor_map.count(tmp_reducer_data->id() + "_1")) {
          ScheduleAssignReduceWithoutLast(ir_sch, node_tensor->name, tmp_reducer_shape, reducer_axes);
          auto tmp_reducer_tensor_0 = tensor_map[tmp_reducer_data->id() + "_0"];
          auto tmp_reducer_loops_0  = ir_sch.GetLoops(tmp_reducer_tensor_0->name);
          auto node_loops           = ir_sch.GetLoops(node_tensor->name);
          if (node_loops.size() < tmp_reducer_loops_0.size()) {
            ir_sch.Split(node_tensor->name, 0, {-1, ir::GetLoopExtent(node_loops[0])});
          }
          CHECK_EQ(ir_sch.GetLoops(node_tensor->name).size(), tmp_reducer_loops_0.size())
              << "node loops and reduce loops must be equal!";
          auto node_block = ir_sch.GetBlock(node_tensor->name);
          ir_sch.SimpleComputeAt(node_block, tmp_reducer_loops_0.back());
        } else {
          OrderAssignReduce(ir_sch, node_tensor->name, reducer_axes);

          auto node_block = ir_sch.GetBlock(node_tensor->name);
          auto node_loops = ir_sch.GetLoops(node_tensor->name);
          if (node_loops.size() < ir_sch.GetLoops(tmp_reducer_tensor->name).size()) {
            ir_sch.Split(node_tensor->name, 0, {-1, ir::GetLoopExtent(node_loops[0])});
          }
          CHECK_EQ(ir_sch.GetLoops(node_tensor->name).size(), ir_sch.GetLoops(tmp_reducer_tensor->name).size())
              << "node loop size and reduce loop size must be equal!";
          ir_sch.SimpleComputeAt(node_block, ir_sch.GetLoops(tmp_reducer_tensor->name).back());
        }
      } else {
        VLOG(2) << "Reduce Schedule for WithLastDimInReduce";
        if (tensor_map.count(reducer_data->id() + "_1")) {
          ScheduleAssignReduceWithLast(ir_sch, node_tensor->name, reducer_shape, reducer_axes);
          auto reducer_1_tensor = tensor_map[reducer_data->id() + "_1"];
          auto reducer_1_block  = ir_sch.GetBlock(reducer_1_tensor->name);

          auto node_loops = ir_sch.GetLoops(node_tensor->name);
          if (ir_sch.GetLoops(node_tensor->name).size() < ir_sch.GetLoops(reducer_1_block).size()) {
            ir_sch.Split(node_tensor->name, 0, {-1, ir::GetLoopExtent(node_loops[0])});
          }
          CHECK_EQ(ir_sch.GetLoops(node_tensor->name).size(), ir_sch.GetLoops(reducer_1_block).size())
              << "node loop size and reduce loop size must be equal!";
          auto node_block      = ir_sch.GetBlock(node_tensor->name);
          auto reducer_1_loops = ir_sch.GetLoops(reducer_1_block);
          ir_sch.SimpleComputeAt(node_block, reducer_1_loops.back());
        } else {
          auto reducer_0_tensor = tensor_map[reducer_data->id() + "_0"];
          auto reducer_0_block  = ir_sch.GetBlock(reducer_0_tensor->name);
          auto reducer_0_loops  = ir_sch.GetLoops(reducer_0_block);

          auto node_loops = ir_sch.GetLoops(node_tensor->name);
          if (node_loops.size() < reducer_0_loops.size()) {
            ir_sch.Split(node_tensor->name, 0, {-1, ir::GetLoopExtent(node_loops[0])});
          }
          CHECK_EQ(ir_sch.GetLoops(node_tensor->name).size(), reducer_0_loops.size())
              << "node loop size and reduce loop size must be equal!";
          auto node_block = ir_sch.GetBlock(node_tensor->name);
          ir_sch.SimpleComputeAt(node_block, reducer_0_loops.back());
        }
      }
      continue;
    }
    // others elemenwise internal node use compute-inline
    auto node_block = ir_sch.GetBlock(node_tensor->name);
    ir_sch.ComputeInline(node_block);
  }
}

std::vector<ir::LoweredFunc> OpLowerer::IRLowerOpaqueOp(GroupPtr& group) {
  VLOG(3) << "LowerOpaqueOp Group : " << group->group_id;
  // get input tensor and output tensor
  CHECK(group->nodes.size() || group->fused_sub_groups.size());
  auto& cinn_strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  auto node = group->fused_sub_groups.size() ? group->fused_sub_groups[0]->nodes.front() : group->nodes.front();
  VLOG(3) << "GetOpFunc of op " << node->id();
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;

  std::vector<ir::Argument> args;
  for (auto& i : node->inlinks_in_order(true)) {
    std::string id = i->source()->as<NodeData>()->id();
    auto shape     = shape_dict_.at(id);
    Type dtype     = type_dict_.at(id);
    CHECK(dtype == Float(32) || dtype.is_bool() || dtype == Int(32))
        << "The dtype of node " << id << " is not float or bool or int! Other dtype is not implemented yet.";
    ir::Tensor input;
    if (dtype == Float(32)) {
      input = lang::Placeholder<float>(id, shape);
    } else if (dtype.is_bool()) {
      input = lang::Placeholder<bool>(id, shape);
    } else if (dtype == Int(32)) {
      input = lang::Placeholder<int>(id, shape);
    }
    inputs.push_back(input);
    cinn_inputs.push_back(common::CINNValue(input));
    group->input_names.push_back(id);
    args.emplace_back(input->buffer, ir::Argument::IO::kInput);
  }

  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;

  auto node_datas = GetAllNodeData(node);
  for (auto node_data : node_datas) {
    VLOG(3) << "cinn_inputs.push_back " << node_data->id();
    group->output_names.push_back(node_data->id());
    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));
    cinn_inputs.push_back(common::CINNValue(node_data->id()));
  }

  auto impl = OpStrategy::SelectImpl(cinn_strategy[node->op()](node->attrs, inputs, out_types, out_shapes, target_));
  common::CINNValuePack pack = impl->fcompute(common::CINNValuePack{cinn_inputs});

  for (int i = 0; i < pack->size() - 1; i++) {
    ir::Expr temp = pack[i];
    // checkout whether the tensor is with buffer.
    if (!temp.as_tensor_ref()->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) {
      inputs.push_back(temp.as_tensor_ref());
      temp.as_tensor_ref()->WithBuffer();
      args.emplace_back(temp.as_tensor_ref()->buffer, ir::Argument::IO::kOutput);
    }
  }

  poly::StageMap stages = pack.back();
  auto func             = lang::LowerVec(group->GetFuncName(), stages, inputs, {}, {}, nullptr, this->target_, true);

  std::vector<common::CINNValue> schedule_inputs;
  // collect tensor
  for (int idx = 0; idx < pack.size() - 1; ++idx) {
    CHECK(pack[idx].is_tensor());
    schedule_inputs.push_back(common::CINNValue(pack[idx]));
  }
  for (auto& f : func) {
    schedule_inputs.push_back(common::CINNValue(f->body));
  }
  // do ast tree schedule
  common::CINNValuePack expr_pack = impl->fschedule(common::CINNValuePack{schedule_inputs});

  std::vector<ir::LoweredFunc> res;
  for (int i = 0; i < expr_pack.size(); i++) {
    ir::Expr func_body = expr_pack[0];
    auto temp_buffers  = lang::GetTempBuffers(inputs, stages, func_body);
    auto function      = ir::_LoweredFunc_::Make(group->GetFuncName(), args, func_body, temp_buffers);
    function->PrepareBufferCastExprs();
    res.push_back(function);
  }
  for (auto& i : res) {
#ifdef CINN_WITH_CUDA
    optim::OptimizeExprGPU(&(i->body));
#endif
    i = optim::Optimize(Expr(i), target_, false).as_lowered_func_ref();
  }
  return res;
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
    VLOG(3) << "ReduceCompute tensor_inputs size is : " << tensor_inputs.size();
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

  auto WithoutLastDimInReduce = [](const std::vector<int>& inshape, std::vector<int>& axes) {
    // if last axis is in reduce.
    axes = axes.empty() ? inshape : axes;
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
      [this, OrderAssignReduce](poly::Stage* stage, const std::vector<int>& inshape, std::vector<int>& axes) {
        axes                = axes.empty() ? inshape : axes;
        int lane            = 1;
        int max_num_threads = this->target_.max_num_threads();
        for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
          lane *= inshape[idx];
        }
        CHECK_LE(lane, max_num_threads / 2) << "Parallel threads must less equal max_num_threads/2 on gpu!";
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

  auto ScheduleAssignReduceWithLast = [this, OrderAssignReduce](
                                          poly::Stage* stage, const std::vector<int>& inshape, std::vector<int>& axes) {
    // find first reduce and second reduce axis.
    axes                 = axes.empty() ? inshape : axes;
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
  CHECK(master_reducer->attrs.attr_store.count("dim"));
  auto master_reducer_axes = absl::get<std::vector<int>>(master_reducer->attrs.attr_store.at("dim"));
  CHECK(master_reducer->inlinks_in_order().size());
  CHECK(this->shape_dict_.count(master_reducer->inlinks_in_order()[0]->source()->id()));
  auto master_reducer_shape = this->shape_dict_.at(master_reducer->inlinks_in_order()[0]->source()->id());

  if (master_reducer_axes.empty()) {
    for (int i = 0; i < master_reducer_shape.size(); ++i) {
      master_reducer_axes.emplace_back(i);
    }
  }

  bool reduce_with_same_shape = true;
  bool without_last_dim       = WithoutLastDimInReduce(master_reducer_shape, master_reducer_axes);
  if (without_last_dim) {
    // check each reduce has same input shape.
    for (auto reducer : group->master_nodes) {
      if (op_pattern_dict[reducer->op()] != framework::kCommReduce) {
        continue;
      }
      if (this->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id()) != master_reducer_shape) {
        reduce_with_same_shape = false;
        break;
      }
    }
  }
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
      if (without_last_dim) {
        // compute at last dimension
        if (node == master_reducer) {
          stage->SimpleComputeAt(master_stage, master_stage->n_out_dims() - 1);
        } else {
          // if don't use block shuffle reduce.
          if (!tensor_map.count(node_data->id() + "_1")) {
            if (reduce_with_same_shape) {
              if (master_reducer_stage->n_out_dims() > 1) {
                stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - 1);
              }
            } else {
              int num_reduce_axis = master_reducer_stage->tensor()->reduce_axis.size();
              if (master_reducer_stage->n_out_dims() > num_reduce_axis) {
                stage->SimpleComputeAt(master_reducer_stage, master_reducer_stage->n_out_dims() - num_reduce_axis - 1);
              }
            }
          } else {
            auto stage_1 = stages[tensor_map[node_data->id() + "_0"]];
            auto stage_2 = stages[tensor_map[master_reducer_data->id() + "_0"]];
            // compute at master reducer
            if (reduce_with_same_shape) {
              stage_1->SimpleComputeAt(stage_2, stage_2->n_out_dims() - 1);
            } else {
              int num_reduce_axis = stage_2->tensor()->reduce_axis.size();
              stage_1->SimpleComputeAt(stage_2, stage_2->n_out_dims() - num_reduce_axis - 1);
            }
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
      if (without_last_dim) {
        VLOG(3) << "Reduce Schedule for WithoutLastDimInReduce";
        auto reducer_stage = master_reducer_stage;
        auto reducer_shape = master_reducer_shape;
        auto reducer_data  = master_reducer_data;
        auto node_shape    = this->shape_dict_.at(node_data->id());

        if (!reduce_with_same_shape) {
          // find reducer for current node to assign
          GroupPtr reducer_group = sub_group;
          if (sub_group->op_pattern_kind != framework::kCommReduce) {
            for (auto& consumer : sub_group->consumer_groups) {
              if (!consumer->belong_groups.count(group)) {
                continue;
              }
              if (consumer->op_pattern_kind == framework::kCommReduce) {
                reducer_group = consumer;
                break;
              }
            }
          }

          if (reducer_group->op_pattern_kind == framework::kCommReduce) {
            for (auto reducer : reducer_group->master_nodes) {
              if (op_pattern_dict[reducer->op()] == framework::kCommReduce) {
                reducer_shape = this->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());
                if (node_shape == reducer_shape) {
                  reducer_data  = GetNodeData(reducer);
                  reducer_stage = stages[tensor_map[reducer_data->id()]];
                  break;
                }
              }
            }
          } else {
            for (auto reducer : group->master_nodes) {
              if (op_pattern_dict[reducer->op()] == framework::kCommReduce) {
                reducer_shape = this->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());
                if (node_shape == reducer_shape) {
                  reducer_data  = GetNodeData(reducer);
                  reducer_stage = stages[tensor_map[reducer_data->id()]];
                  break;
                }
              }
            }
          }
        }
        CHECK(node_shape == reducer_shape);

        // if used block shuffle reduce
        if (tensor_map.count(reducer_data->id() + "_1")) {
          ScheduleAssignReduceWithoutLast(stage, reducer_shape, master_reducer_axes);
          auto stage_0 = stages[tensor_map[reducer_data->id() + "_0"]];
          if (stage->n_out_dims() < stage_0->n_out_dims()) {
            stage->Split(0, stage->GetDimRange(0));
          }
          CHECK_EQ(stage->n_out_dims(), stage_0->n_out_dims()) << "stage and stage_0's n_out_dims must be equal!";
          if (reduce_with_same_shape) {
            stage->SimpleComputeAt(stage_0, stage_0->n_out_dims() - 1);
          } else {
            int num_reduce_axis = stage_0->tensor()->reduce_axis.size();
            stage->SimpleComputeAt(stage_0, stage_0->n_out_dims() - num_reduce_axis - 1);
          }
        } else {
          OrderAssignReduce(stage, master_reducer_axes);
          if (stage->n_out_dims() < reducer_stage->n_out_dims()) {
            stage->Split(0, stage->GetDimRange(0));
          }
          CHECK_EQ(stage->n_out_dims(), reducer_stage->n_out_dims())
              << "stage and master_reducer_stage's n_out_dims must be equal!";
          if (reduce_with_same_shape) {
            stage->SimpleComputeAt(reducer_stage, reducer_stage->n_out_dims() - 1);
          } else {
            int num_reduce_axis = reducer_stage->tensor()->reduce_axis.size();
            stage->SimpleComputeAt(reducer_stage, reducer_stage->n_out_dims() - num_reduce_axis - 1);
          }
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
  CHECK(group->nodes.size() || group->fused_sub_groups.size());
  auto& cinn_strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  auto node = group->fused_sub_groups.size() ? group->fused_sub_groups[0]->nodes.front() : group->nodes.front();
  std::vector<ir::Tensor> tensor_inputs;
  std::vector<common::CINNValue> cinn_inputs;
  for (auto& link : node->inlinks_in_order(true)) {
    auto source = link->source();
    CHECK(source);
    auto source_data = source->safe_as<NodeData>();
    CHECK(source_data);

    auto id    = source_data->id();
    auto shape = this->shape_dict_.at(id);
    auto dtype = this->type_dict_.at(id);

    ir::Tensor tensor;
    if (dtype == Float(32)) {
      tensor = lang::Placeholder<float>(id, shape);
    } else if (dtype.is_bool()) {
      tensor = lang::Placeholder<bool>(id, shape);
    } else if (dtype == Int(32)) {
      tensor = lang::Placeholder<int>(id, shape);
    }
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

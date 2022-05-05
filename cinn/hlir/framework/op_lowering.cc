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

std::vector<NodeData*> GetAllNodeData(Node* node) {
  std::vector<NodeData*> node_datas;
  for (auto& link : node->outlinks()) {
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
  VLOG(11) << "Lowering Group : " << group->group_id << " , Op Pattern : " << group->op_pattern_kind;
  switch (group->op_pattern_kind) {
    case framework::kElemWise:
    case framework::kBroadcast:
    case framework::kInjective:
      return LowerOp(&OpLowerer::ElementwiseCompute, &OpLowerer::ElementwiseSchedule, group);
    case framework::kCommReduce:
      LOG(FATAL) << "Group Pattern Kind Is ReduceCompute!";
      // return LowerOp(&OpLowerer::ReduceCompute, &OpLowerer::ReduceSchedule, group);
    case framework::kOutEWiseFusable:
      LOG(FATAL) << "Group Pattern Kind Is OutEWiseFusableCompute!";
      // return LowerOp(&OpLowerer::OutEWiseFusableCompute, &OpLowerer::OutEWiseFusableSchedule, group);
    case framework::kOpaque:
      LOG(FATAL) << "Group Pattern Kind Is LowerOpaqueOp!";
      // return LowerOpaqueOp(group);
    default:
      LOG(FATAL) << "Group Pattern Kind Is Unknown!";
  }
}

// fusion op lowering
std::vector<ir::LoweredFunc> OpLowerer::LowerOp(ComputeFunction compute, ScheduleFunction schedule, GroupPtr& group) {
  poly::StageMap stages;
  std::vector<ir::Tensor> func_tensors;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  std::vector<ir::IRSchedule> ir_schedules;
  // do compute.
  if (group->fused_sub_groups.size() == 0) {
    ir_schedules.push_back((this->*compute)(stages, func_tensors, tensor_map, group, group));
  } else {
    for (auto& sub_group : group->fused_sub_groups) {
      ir_schedules.push_back((this->*compute)(stages, func_tensors, tensor_map, group, sub_group));
    }
  }

  // do schedule.
  if (group->fused_sub_groups.size() == 0) {
    (this->*schedule)(ir_schedules[0], stages, tensor_map, group, group);
  } else {
    int i = 0;
    for (auto& sub_group : group->fused_sub_groups) {
      (this->*schedule)(ir_schedules[i], stages, tensor_map, group, sub_group);
      i++;
    }
  }
  LOG(INFO) << "After compute and schedule.";
  CHECK_EQ(ir_schedules.size(), 1);
  for (auto& args : func_tensors) {
    // input node data name.
    group->input_names.push_back(args->name);
  }

  auto func_input_tensors = func_tensors;
  LOG(INFO) << "LowerOp stage 2";
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
      if (!tensor->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) {
        func_tensors.push_back(tensor);
      }
      // update post
      post = "_" + std::to_string(idx);
    }
  }
  LOG(INFO) << "LowerOp stage 3";
  ir::LoweredFunc func;
  auto func_body = ir_schedules[0].GetModule().GetExprs().at(0);
  LOG(INFO) << "func_body is : " << func_body;
  auto temp_buffers = lang::GetTempBuffers(func_tensors, stages, func_body);
  LOG(INFO) << "LowerOp stage 3";
  auto func_args = lang::GetArgs(func_body, func_input_tensors);
  LOG(INFO) << "LowerOp stage 4";
  func = ir::_LoweredFunc_::Make(
      func_name_prefix + group->group_id, func_args, ir_schedules[0].GetModule().GetExprs().at(0), temp_buffers);
  LOG(INFO) << "LowerOp stage 5";
  func->PrepareBufferCastExprs();
  LOG(INFO) << "LowerOp stage 6";
  optim::OptimizeExprGPU(&(func->body));
  // i->body = optim::Optimize(i->body, target_, false);
  LOG(INFO) << "LowerOp stage 7";
  func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();
  LOG(INFO) << "LowerOp stage 8";
  return {func};
  // return lang::LowerVec(func_name_prefix + group->group_id, stages, func_args, {}, {}, nullptr, this->target_);
}

std::vector<ir::Tensor> OpLowerer::CollectInputTensor(std::vector<ir::Tensor>& func_tensors,
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
      func_tensors.push_back(tensor);
    }
  }

  return tensor_inputs;
}

ir::IRSchedule OpLowerer::ElementwiseCompute(poly::StageMap& stages,
                                             std::vector<ir::Tensor>& func_tensors,
                                             std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                             const GroupPtr& group,
                                             const GroupPtr& sub_group) {
  VLOG(11) << "ElementwiseCompute Group : " << sub_group->group_id;
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  std::vector<std::vector<common::CINNValue>> res;
  int index = 0;
  for (auto& node : sub_group->nodes) {
    LOG(INFO) << "Begin node with index : " << index;
    auto node_data = GetNodeData(node);
    LOG(INFO) << "Its node_data name is : " << node_data->id();
    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(func_tensors, tensor_map, node));
    for (auto& tensor : tensor_inputs) {
      stages->InsertLazily(tensor);
      cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    }
    cinn_inputs.push_back(common::CINNValue(node_data->id().c_str()));

    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;

    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));

    auto impl =
        OpStrategy::SelectImpl(strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, this->target_));
    // do compute
    common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});

    poly::StageMap stages = C.back();
    // make sure all the tensors in the stages before schedule launch.
    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      stages->InsertLazily(temp.as_tensor_ref());
    }
    std::vector<common::CINNValue> schedule_inputs;
    schedule_inputs.push_back(common::CINNValue(C.back()));

    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      // checkout whether the tensor is with buffer.
      if (!temp.as_tensor_ref()->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) {
        tensor_inputs.push_back(temp.as_tensor_ref());
      }
    }
    auto func = lang::LowerVec("fn_" + node->id(), stages, tensor_inputs, {}, {}, nullptr, this->target_, true);

    for (int i = func.size() - 1; i >= 0; i--) {
      auto ast_expr = func[i]->body;
      schedule_inputs.insert(schedule_inputs.begin(), common::CINNValue(ast_expr));
    }

    // if (group->master_nodes.count(node)) {
    // do shedule
    LOG(INFO) << "Node with index : " << index << " is master node";
    Expr print_expr = schedule_inputs[0];
    LOG(INFO) << "Before Schedule, AST is : " << print_expr;
    common::CINNValuePack expr_pack = impl->fschedule(common::CINNValuePack{schedule_inputs});
    Expr print_expr2                = schedule_inputs[0];
    LOG(INFO) << "After Schedule, AST is : " << print_expr2;
    std::vector<common::CINNValue> new_schedule_inputs;
    for (int i = 0; i < expr_pack.size(); i++) new_schedule_inputs.push_back(schedule_inputs[i]);
    new_schedule_inputs.push_back(schedule_inputs.back());
    schedule_inputs = new_schedule_inputs;
    //}

    if (group->master_nodes.count(node)) {
      // do shedule
      LOG(INFO) << "Node with index : " << index << " is master node";
    }

    res.push_back(schedule_inputs);
    index++;
  }
  std::vector<Expr> vec_ast;
  int i = 0;
  for (auto& arg_pack : res) {
    for (int i = 0; i < arg_pack.size() - 1; i++) {
      Expr temp = arg_pack[i];
      vec_ast.push_back(temp);
    }
  }
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();

  return ir_sch;
}

void OpLowerer::ElementwiseSchedule(ir::IRSchedule& ir_sch,
                                    poly::StageMap& stages,
                                    std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                    const GroupPtr& group,
                                    const GroupPtr& sub_group) {
  LOG(INFO) << "ElementwiseSchedule Group : " << sub_group->group_id;
  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);
    LOG(INFO) << "Begin node : " << node_data->id();
    // CHECK_EQ(tensor_map[node_data->id()]->name, node_data->id());
    // if group master node
    if (group->master_nodes.count(node)) {
      LOG(INFO) << "This node is master node.";
      continue;
    }

    // if node is fringe node or internal node, fringe node is output node of sub-graph
    if (group->output_nodes.count(node) || group->internal_nodes.count(node) || sub_group->internal_nodes.count(node)) {
      LOG(INFO) << "Set node's buffer to be local";
      /*       auto master_node_data = GetNodeData(*group->master_nodes.begin());
            auto master_node_stage = stages[tensor_map[master_node_data->id()]];
            auto node_stage        = stages[tensor_map[node_data->id()]]; */
      // copy schedule from master node
      // node_stage->CopyTransform(master_node_stage);
      // node_stage->CopyLoopInfo(master_node_stage);
      // internal node use buffer
      if (group->internal_nodes.count(node) || sub_group->internal_nodes.count(node)) {
        LOG(INFO) << "node is internal node";
        auto tensor_block = ir_sch.GetBlock(node_data->id());
        LOG(INFO) << "node is internal node2";
        ir_sch.SetBuffer(tensor_block, "local");
        LOG(INFO) << "node is internal node3";
        // node_stage->SetBuffer("local");
      }
      // compute at master node
      // node_stage->SimpleComputeAt(master_node_stage, master_node_stage->n_out_dims() - 1);
      continue;
    }
    LOG(INFO) << "Get block of node";

    // others elemenwise internal node use compute-inline
    auto tensor_block = ir_sch.GetBlock(node_data->id());
    LOG(INFO) << "Before setting node to be computeinline";
    ir_sch.ComputeInline(tensor_block);
    LOG(INFO) << "After setting node to be computeinline";

    // stages[tensor_map[node_data->id()]]->ComputeInline();
  }
  LOG(INFO) << "After group scheduling, AST is: " << ir_sch.GetModule().GetExprs().at(0);
}

std::vector<ir::LoweredFunc> OpLowerer::LowerOpaqueOp(GroupPtr& group) {
  VLOG(11) << "LowerOpaqueOp Group : " << group->group_id;
  // get input tensor and output tensor
  std::vector<ir::Tensor> func_args;
  CHECK_EQ(group->nodes.size(), 1) << "fusion op exist more than 1 op.";

  auto node = *group->nodes.begin();

  auto& strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;
  std::vector<std::vector<int>> output_shapes;
  std::vector<ir::Tensor> input_args;
  VLOG(3) << "GetOpFunc of op " << node->id();
  for (auto& i : node->inlinks_in_order(true)) {
    std::string input_id = i->source()->as<NodeData>()->id();
    auto in_shape        = shape_dict.at(input_id);
    Type dtype           = dtype_dict.at(input_id);
    CHECK(dtype == Float(32) || dtype.is_bool() || dtype == Int(32))
        << "The dtype of node " << input_id << " is not float or bool or int! Other dtype is not implemented yet.";
    ir::Tensor temp;
    if (dtype == Float(32)) {
      temp = lang::Placeholder<float>(input_id, in_shape);
    } else if (dtype.is_bool()) {
      temp = lang::Placeholder<bool>(input_id, in_shape);
    } else if (dtype == Int(32)) {
      temp = lang::Placeholder<int>(input_id, in_shape);
    }
    input_args.push_back(temp);
    inputs.push_back(temp);
    cinn_inputs.push_back(common::CINNValue(temp));
  }
  std::vector<Type> out_types;
  for (auto& out : node->outlinks_in_order(true)) {
    std::string out_id = out->sink()->safe_as<NodeData>()->id();
    auto out_shape     = shape_dict.at(out_id);
    Type dtype         = dtype_dict.at(out_id);
    output_shapes.push_back(out_shape);
    out_types.push_back(dtype);
  }

  auto impl = OpStrategy::SelectImpl(strategy[node->op()](node->attrs, inputs, out_types, output_shapes, target_));

  common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});
  poly::StageMap stages   = C.back();
  // make sure all the tensors in the stages before schedule launch.
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    stages->InsertLazily(temp.as_tensor_ref());
  }
  std::vector<common::CINNValue> schedule_inputs;
  schedule_inputs.push_back(common::CINNValue(C.back()));
  // C = impl->fschedule(C);
  auto inputs_arg = inputs;
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    // checkout whether the tensor is with buffer.
    if (!temp.as_tensor_ref()->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) {
      inputs.push_back(temp.as_tensor_ref());
    }
  }

  auto func =
      lang::LowerVec(GetOrGenFullFuncName(GenOpFuncName(node)), stages, inputs, {}, {}, nullptr, this->target_, true);
  for (int i = 0; i < func.size(); i++) {
    LOG(INFO) << "Before schedule, func[" << i << "] is : " << func[i];
  }
  // CHECK_EQ(func.size(), 1UL);
  for (int i = func.size() - 1; i >= 0; i--) {
    auto ast_expr = func[i]->body;
    schedule_inputs.insert(schedule_inputs.begin(), common::CINNValue(ast_expr));
  }

  common::CINNValuePack expr_pack = impl->fschedule(common::CINNValuePack{schedule_inputs});

  {
    ir::Expr temp = C[0];
    if (!temp.as_tensor_ref()->buffer.defined() || this->target_ != common::DefaultNVGPUTarget() ||
        temp.as_tensor_ref()->buffer->memory_type == ir::MemoryType::Heap) {
      if (!temp.as_tensor_ref()->buffer.defined()) LOG(INFO) << temp.as_tensor_ref()->name << " buffer is not defined.";
      if (this->target_ != common::DefaultNVGPUTarget()) LOG(INFO) << "target is not nvgpu!";
      if (temp.as_tensor_ref()->buffer->memory_type == ir::MemoryType::Heap) LOG(INFO) << "buffer memory type is Heap!";
      LOG(INFO) << "inputs_arg push back " << temp.as_tensor_ref()->name
                << " with buffer name : " << temp.as_tensor_ref()->buffer->name << " with mem type "
                << temp.as_tensor_ref()->buffer->memory_type;
      inputs_arg.push_back(temp.as_tensor_ref());
    }
  }

  VLOG(3) << "expr_pack.size() is : " << expr_pack.size();
  std::vector<ir::LoweredFunc> res;
  for (int i = 0; i < expr_pack.size(); i++) {
    auto new_args      = lang::GetArgs(func[i]->body, input_args);
    func[i]->args      = new_args;
    auto temp_buffers  = lang::GetTempBuffers(inputs_arg, stages, func[i]->body);
    func[i]->temp_bufs = temp_buffers;
    func[i]->PrepareBufferCastExprs();
    res.push_back(func[i]);
  }
  for (auto& i : res) {
    optim::OptimizeExprGPU(&(i->body));
    // i->body = optim::Optimize(i->body, target_, false);
    i = optim::Optimize(Expr(i), target_, false).as_lowered_func_ref();
    LOG(INFO) << "res[i]'s name is : " << i->name;
  }
  return res;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

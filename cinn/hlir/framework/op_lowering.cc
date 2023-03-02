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

#include "cinn/hlir/framework/op_lowering_util.h"
#include "cinn/hlir/op/external_api_registry.h"
#include "cinn/optim/transform_gpu_forloop.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace framework {

using common::float16;

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
using cinn::hlir::op::ExternalApiRegistry;

OpLowerer::OpLowerer(const absl::flat_hash_map<std::string, Type>& type_dict,
                     const absl::flat_hash_map<std::string, shape_t>& shape_dict,
                     const Target& target)
    : type_dict_(type_dict), shape_dict_(shape_dict), target_(target) {}

std::vector<ir::LoweredFunc> OpLowerer::LowerWithoutSchedule(GroupPtr& group) {
  VLOG(3) << "Lowering Group : " << group->group_id << " , Op Pattern : " << group->op_pattern_kind;
  if (FLAGS_cinn_ir_schedule) {
    switch (group->op_pattern_kind) {
      case framework::kElementWise:
      case framework::kBroadcast:
      case framework::kInjective:
        return IRLowerOpWithoutSchedule(&OpLowerer::IRElementwiseCompute, group);
      case framework::kReduction:
        return IRLowerOpWithoutSchedule(&OpLowerer::IRReduceCompute, group);
      case framework::kOutFusible:
        LOG(FATAL) << "Group Pattern Kind kOutFusible Is Not Implemented!";
      case framework::kNonFusible:
        return IRLowerNonFusibleOp(group, /*apply_impl_schedule = */ false);
      default:
        LOG(FATAL) << "Group Pattern Kind kNonFusible Is Not Implemented!";
    }
  } else {
    LOG(FATAL) << "Previous IR Schedule Is Unsupport Now!";
  }
}

std::vector<ir::LoweredFunc> OpLowerer::Lower(GroupPtr& group) {
  VLOG(3) << "Lowering Group : " << group->group_id << " , Op Pattern : " << group->op_pattern_kind;
  if (FLAGS_cinn_ir_schedule) {
    switch (group->op_pattern_kind) {
      case framework::kElementWise:
      case framework::kBroadcast:
      case framework::kInjective:
        return IRLowerOp(&OpLowerer::IRElementwiseCompute, group);
      case framework::kReduction:
        return IRLowerOp(&OpLowerer::IRReduceCompute, group);
      case framework::kOutFusible:
        LOG(FATAL) << "Group Pattern Kind kOutFusible Is Not Implemented!";
      case framework::kNonFusible:
        return IRLowerNonFusibleOp(group, /*apply_impl_schedule = */ true);
      default:
        LOG(FATAL) << "Group Pattern Kind Is Unknown!";
    }
  } else {
    LOG(FATAL) << "Previous IR Schedule Is Unsupport Now!";
  }
}

std::vector<ir::LoweredFunc> OpLowerer::IRLowerOpWithoutSchedule(IRComputeFunction compute, GroupPtr& group) {
  poly::StageMap stages;
  std::vector<ir::Tensor> arg_tensors;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  // do compute.
  VLOG(3) << "group->fused_sub_groups.size() is : " << group->fused_sub_groups.size();
  std::vector<Expr> ast_exprs;
  if (group->fused_sub_groups.size() == 0) {
    ast_exprs = (this->*compute)(stages, arg_tensors, tensor_map, group, group, /*apply_impl_schedule = */ false);
  } else {
    for (auto& sub_group : group->fused_sub_groups) {
      auto exprs =
          (this->*compute)(stages, arg_tensors, tensor_map, group, sub_group, /*apply_impl_schedule = */ false);
      ast_exprs.insert(ast_exprs.end(), exprs.begin(), exprs.end());
    }
  }
  ir::ModuleExpr mod_expr(ast_exprs);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();

  VLOG(3) << "After IRLowerOp compute, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  // function args
  group->input_names.clear();
  std::vector<ir::Argument> func_args;
  for (auto& args : arg_tensors) {
    // input node data name.
    group->input_names.push_back(args->name);
    // input args
    func_args.emplace_back(args->buffer, ir::Argument::IO::kInput);
  }

  group->output_names.clear();
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

  std::unordered_set<std::string> args_map;
  for (auto arg : func_args) {
    args_map.insert(arg.name());
  }

  for (auto& tensor : tensor_map) {
    if (args_map.count("_" + tensor.first)) {
      continue;
    }
    arg_tensors.push_back(tensor.second);
    // use the underlying tensor name to be consistent with the argument name in the lowered function
    group->output_names.push_back(tensor.second->name);
    func_args.emplace_back(tensor.second->buffer, ir::Argument::IO::kOutput);
  }

  auto func_body = ir_sch.GetModule().GetExprs().at(0);
#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&(func_body));
#endif

  auto temp_buffers = lang::GetTempBuffers(arg_tensors, stages, func_body);
  auto func =
      ir::_LoweredFunc_::Make(group->GetFuncName(), func_args, ir_sch.GetModule().GetExprs().at(0), temp_buffers);
  func->PrepareBufferCastExprs();
  func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();

  return {func};
}

std::vector<ir::LoweredFunc> OpLowerer::IRLowerOp(IRComputeFunction compute, GroupPtr& group) {
  poly::StageMap stages;
  std::vector<ir::Tensor> arg_tensors;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  // do compute.
  VLOG(3) << "group->fused_sub_groups.size() is : " << group->fused_sub_groups.size();
  std::vector<Expr> ast_exprs;
  if (group->fused_sub_groups.size() == 0) {
    ast_exprs = (this->*compute)(stages, arg_tensors, tensor_map, group, group, /*apply_impl_schedule = */ true);
  } else {
    for (auto& sub_group : group->fused_sub_groups) {
      auto exprs = (this->*compute)(stages, arg_tensors, tensor_map, group, sub_group, /*apply_impl_schedule = */ true);
      ast_exprs.insert(ast_exprs.end(), exprs.begin(), exprs.end());
    }
  }
  ir::ModuleExpr mod_expr(ast_exprs);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();

  Node* first  = nullptr;
  Node* second = nullptr;
  // do schedule.
  VLOG(3) << "Before IRLowerOp schedule, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  /*
  if (group->fused_sub_groups.size() == 0) {
    (this->*schedule)(ir_sch, tensor_map, group, group, first, second);
  } else {
    // do schedule from back to front.
    for (int idx = group->fused_sub_groups.size() - 1; idx >= 0; --idx) {
      (this->*schedule)(ir_sch, tensor_map, group, group->fused_sub_groups[idx], first, second);
    }
  }
  */
  IRSchedule(ir_sch, group, tensor_map);
  VLOG(3) << "After IRLowerOp schedule, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  // function args
  group->input_names.clear();
  std::vector<ir::Argument> func_args;
  for (auto& args : arg_tensors) {
    // input node data name.
    group->input_names.push_back(args->name);
    // input args
    func_args.emplace_back(args->buffer, ir::Argument::IO::kInput);
  }

  group->output_names.clear();
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
  auto func_body = ir_sch.GetModule().GetExprs().at(0);
#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&(func_body));
#endif

  auto temp_buffers = lang::GetTempBuffers(arg_tensors, stages, func_body);
  auto func =
      ir::_LoweredFunc_::Make(group->GetFuncName(), func_args, ir_sch.GetModule().GetExprs().at(0), temp_buffers);
  func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();
  return {func};
}

std::vector<ir::Tensor> OpLowerer::CollectInputTensor(const Node* node,
                                                      std::vector<ir::Tensor>& func_args,
                                                      std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  std::vector<ir::Tensor> tensors;
  // get all input nodes
  for (auto& node_data : GetProducerNodeData(node)) {
    CHECK(node_data);
    auto tensor = GetTensor(node_data, this->type_dict_, this->shape_dict_);
    if (!tensor_map.count(node_data->id())) {
      tensor_map[node_data->id()] = tensor;
      // record func input args
      func_args.push_back(tensor);
    }
    tensors.push_back(tensor);
  }
  return tensors;
}

std::vector<Expr> OpLowerer::IRElementwiseCompute(poly::StageMap& stages,
                                                  std::vector<ir::Tensor>& func_tensors,
                                                  std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                                  const GroupPtr& group,
                                                  const GroupPtr& sub_group,
                                                  bool apply_impl_schedule) {
  VLOG(2) << "ElementwiseCompute Group : " << sub_group->group_id;
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  std::vector<Expr> ast_exprs;
  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);
    CHECK_EQ(GetAllNodeData(node).size(), 1U);
    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(node, func_tensors, tensor_map));
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

    if (apply_impl_schedule) {
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
    } else {
      ast_exprs.push_back(func[0]->body);
    }
  }

  return ast_exprs;
}

std::vector<Expr> OpLowerer::IRReduceCompute(poly::StageMap& stages,
                                             std::vector<ir::Tensor>& func_args,
                                             std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                             const GroupPtr& group,
                                             const GroupPtr& sub_group,
                                             bool apply_impl_schedule) {
  VLOG(2) << "ReduceCompute Group : " << sub_group->group_id;
  auto& cinn_strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  std::vector<Expr> ast_exprs;
  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);
    VLOG(3) << "In ReduceCompute, process node: " << node->id() << " with op type: " << node->op()->name;

    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(node, func_args, tensor_map));
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

    // node is kReduction
    if (op_pattern_dict[node->op()] == framework::kReduction && apply_impl_schedule) {
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

std::vector<ir::LoweredFunc> OpLowerer::IRLowerNonFusibleOp(GroupPtr& group, bool apply_impl_schedule) {
  VLOG(3) << "LowerNonFusibleOp Group : " << group->group_id;
  // get input tensor and output tensor
  CHECK(group->nodes.size() || group->fused_sub_groups.size());
  auto& cinn_strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  auto node = group->fused_sub_groups.size() ? group->fused_sub_groups[0]->nodes.front() : group->nodes.front();
  VLOG(3) << "GetOpFunc of op " << node->id();
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;

  std::vector<ir::Argument> args;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  for (auto& node_data : GetProducerNodeData(node)) {
    CHECK(node_data);
    ir::Tensor tensor;
    if (!tensor_map.count(node_data->id())) {
      tensor = GetTensor(node_data, this->type_dict_, this->shape_dict_);
      // record tensor.
      tensor_map[node_data->id()] = tensor;
      // input name.
      group->input_names.push_back(node_data->id());
      // input type.
      args.emplace_back(tensor->buffer, ir::Argument::IO::kInput);
    } else {
      tensor = tensor_map[node_data->id()];
    }
    inputs.push_back(tensor);
    cinn_inputs.push_back(common::CINNValue(tensor));
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
  // if node op is custom_call, apply custom_call compute.
  if (node->op()->name == "custom_call") {
    std::string external_api;
    if (node->attrs.attr_store.count("custom_call")) {
      external_api = absl::get<std::string>(node->attrs.attr_store.at("custom_call"));
    } else {
      external_api = ExternalApiRegistry::Global()->GetExternalApi(node, target_);
    }
    std::vector<common::CINNValue> compute_args = {common::CINNValue(group->GetFuncName()),
                                                   common::CINNValue(external_api)};
    common::CINNValuePack pack                  = impl->fcompute(common::CINNValuePack{compute_args});
    CHECK_EQ(pack.size(), 1UL);
    // reset input names as extern api input args can't be remove duplicate.
    group->input_names.clear();
    for (auto& inode : node->inlinks_in_order(true)) {
      group->input_names.push_back(inode->source()->as<NodeData>()->id());
    }
    return {pack[0].operator ir::Expr().as_lowered_func_ref()};
  }

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

  if (apply_impl_schedule) {
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

    ir::Expr func_body = expr_pack[0];
    std::vector<std::string> input_output_nodes(group->input_names);
    input_output_nodes.insert(input_output_nodes.end(), group->output_names.begin(), group->output_names.end());
    VLOG(6) << "func.size() = " << func.size() << ", expr_pack.size() = " << expr_pack.size();
    VLOG(6) << "args.size() = " << args.size() << ", input_output_nodes.size() = " << input_output_nodes.size();
    if (args.size() > input_output_nodes.size()) {
      args = lang::GetArgs(func_body, input_output_nodes);
    }
    std::vector<ir::LoweredFunc> res;
    for (int i = 0; i < expr_pack.size(); i++) {
      ir::Expr func_body = expr_pack[0];
#ifdef CINN_WITH_CUDA
      optim::OptimizeExprGPU(&(func_body));
#endif
      auto temp_buffers = lang::GetTempBuffers(inputs, stages, func_body);
      auto function     = ir::_LoweredFunc_::Make(group->GetFuncName(), args, func_body, temp_buffers);
      res.push_back(function);
    }
    for (auto& i : res) {
      i = optim::Optimize(Expr(i), target_, false).as_lowered_func_ref();
    }
    return res;
  } else {
    for (auto& f : func) {
#ifdef CINN_WITH_CUDA
      optim::OptimizeExprGPU(&(f->body));
#endif
      f = optim::Optimize(Expr(f), target_, false).as_lowered_func_ref();
    }
    return func;
  }
}

// do compute
void OpLowerer::IRSchedule(ir::IRSchedule& ir_sch,
                           const GroupPtr& group,
                           const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  // topological order.
  std::unordered_set<Node*> nodes_set = group->NodeSet();
  std::vector<Node*> nodes_in_order   = TopologicalOrder(group);
  // find reducer.
  std::unordered_set<Node*> nodes_inline;
  auto greducer         = FindGlobalReducer(nodes_in_order);
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  // do schedule
  for (auto node : nodes_in_order) {
    // consumers.
    auto consumers      = GetConsumersInSet(node, nodes_set);
    const Node* reducer = greducer ? FindNearestReducer(node, nodes_set) : greducer;

    // node can be inline.
    if (CanbeInline(node, consumers, reducer, nodes_in_order.front(), group, nodes_set, this->shape_dict_)) {
      // if exist global reduce node.
      if (greducer) {
        auto loops = ir_sch.GetLoops(GetNodeData(node)->id());
        if (op_pattern_dict[node->op()] == framework::kElementWise) {
          ir_sch.FlattenLoops(loops, true);
        } else {
          ir_sch.FlattenLoops(loops, false);
        }
      }

      auto block = ir_sch.GetBlock(GetNodeData(node)->id());
      ir_sch.ComputeInline(block);
      nodes_inline.insert(node);
      continue;
    }
    // find master to computeat.
    auto master = GetMasterToComputeAt(node, nodes_in_order, nodes_inline, nodes_set, this->shape_dict_);

    // assign to reducer/master loop.
    if (reducer) {
      // if node is vertical with reduce, loop assign reducer.
      LoopAssignReduce(ir_sch, node, reducer, this->target_, tensor_map, this->shape_dict_);
    } else if (greducer) {
      // if node is horizontal with reduce or node is reduce, loop assign master.
      auto loops = ir_sch.GetLoops(GetNodeData(node)->id());
      if (op_pattern_dict[node->op()] == framework::kElementWise) {
        ir_sch.FlattenLoops(loops, true);
      } else if (op_pattern_dict[node->op()] == framework::kReduction) {
      } else {
        ir_sch.FlattenLoops(loops, false);
      }
    }
    // do loop fuse.
    LoopComputeAt(ir_sch, node, master ? master : nodes_in_order.front(), group, this->shape_dict_, tensor_map);
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

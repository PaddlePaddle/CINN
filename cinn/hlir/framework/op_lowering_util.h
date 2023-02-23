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

#include <queue>

#include "cinn/hlir/framework/op_lowering.h"

namespace cinn {
namespace hlir {
namespace framework {

inline NodeData* GetNodeData(const Node* node) {
  auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
  CHECK(node_data);
  return node_data;
}

inline std::vector<NodeData*> GetAllNodeData(const Node* node) {
  std::vector<NodeData*> node_datas;
  for (auto& link : node->outlinks_in_order(true)) {
    auto node_data = link->sink()->safe_as<NodeData>();
    CHECK(node_data);
    node_datas.push_back(node_data);
  }

  return node_datas;
}

inline std::vector<Node*> GetConsumers(const Node* node) {
  std::vector<Node*> consumers;
  auto node_data = GetNodeData(node);
  for (auto& link : node_data->outlinks()) {
    auto consumer = link->sink()->safe_as<Node>();
    CHECK(consumer);
    consumers.push_back(consumer);
  }
  return consumers;
}

inline std::vector<Node*> GetConsumers(const Node* node, const std::unordered_set<Node*>& node_set) {
  std::vector<Node*> consumers;
  auto node_data = GetNodeData(node);
  for (auto& link : node_data->outlinks()) {
    auto consumer = link->sink()->safe_as<Node>();
    CHECK(consumer);
    if (node_set.count(consumer)) {
      consumers.push_back(consumer);
    }
  }
  return consumers;
}

inline std::vector<Node*> GetProducers(const Node* node) {
  std::vector<Node*> producers;
  for (auto& link : node->inlinks_in_order(true)) {
    auto data = link->source()->safe_as<NodeData>();
    CHECK(data);
    if (data->source_node.get()) {
      producers.push_back(data->source_node.get());
    }
  }
  return producers;
}

inline bool IsConstOp(const framework::Node* node) {
  static std::unordered_set<std::string> const_op_type = {"const_scalar", "fill_constant", "arange"};
  if (const_op_type.count(node->op()->name)) {
    return true;
  } else {
    return false;
  }
}

inline std::vector<int> GetInputShape(const Node* node, const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto producers = GetProducers(node);
  CHECK(producers.size());

  auto producer_data = GetNodeData(producers.front());
  return shape_dict.at(producer_data->id());
}

inline std::vector<int> GetOutputShape(const Node* node, const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto node_data = GetNodeData(node);
  return shape_dict.at(node_data->id());
}

inline std::vector<Node*> TopologicalOrder(const GroupPtr& group) {
  std::vector<Node*> nodes_in_order;
  std::unordered_set<Node*> node_set = group->NodeSet();

  while (!node_set.empty()) {
    auto tmp_node_set = node_set;
    for (auto node : tmp_node_set) {
      auto consumers     = GetConsumers(node, node_set);
      bool cant_be_erase = false;
      for (auto consumer : consumers) {
        if (node_set.count(consumer)) {
          cant_be_erase = true;
          break;
        }
      }

      if (cant_be_erase) continue;
      nodes_in_order.push_back(node);
      node_set.erase(node);
    }
  }

  return nodes_in_order;
}

inline Node* FindReducer(std::vector<Node*> node_in_order) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  for (auto iter = node_in_order.rbegin(); iter != node_in_order.rend(); ++iter) {
    if (op_pattern_dict[(*iter)->op()] == framework::kReduction) {
      return *iter;
    }
  }

  return nullptr;
}

inline bool WithoutLastDimInReduce(const std::vector<int>& shape, const std::vector<int>& axes) {
  if (axes.empty()) {
    return false;
  }
  // if last axis is in reduce.
  if (std::find(axes.begin(), axes.end(), shape.size() - 1) != axes.end() ||
      std::find(axes.begin(), axes.end(), -1) != axes.end()) {
    return false;
  }

  int sum_last_axes = 1;
  for (int idx = axes.back() + 1; idx < shape.size(); ++idx) {
    sum_last_axes *= shape[idx];
  }

  if (sum_last_axes > 1) {
    return true;
  } else {
    return false;
  }
}

inline void LoopOrderAssignReduce(ir::IRSchedule& ir_sch,
                                  const std::string& block_name,
                                  const std::vector<int>& axes,
                                  const common::Target& target,
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

  if (ir::GetLoopExtent(loops[index]) > target.max_num_threads()) {
    ir_sch.Split(block_name, index, {-1, target.max_num_threads()});
  }

  // fuse index - 1 times
  for (int idx = 0; idx < index - 1; ++idx) {
    ir_sch.Fuse(block_name, {0, 1});
  }
}

inline void LoopAssignReduceWithoutLast(ir::IRSchedule& ir_sch,
                                        const std::string& block_name,
                                        const std::vector<int>& inshape,
                                        const std::vector<int>& axes,
                                        const common::Target& target) {
  CHECK(axes.size());
  int lane            = 1;
  int max_num_threads = target.max_num_threads();
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
  LoopOrderAssignReduce(ir_sch, block_name, axes, target);
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
}

inline void LoopAssignReduceWithLast(ir::IRSchedule& ir_sch,
                                     const std::string& block_name,
                                     const std::vector<int>& inshape,
                                     const std::vector<int>& axes,
                                     const common::Target& target) {
  // find first reduce and second reduce axis.
  int lane             = 1;
  int index            = static_cast<int>(axes.size()) - 1;
  auto max_num_threads = target.max_num_threads();
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
    LoopOrderAssignReduce(ir_sch, block_name, first_axes, target);
  } else {
    int fuse_times = axes.size() - (index + 1) - 1;
    for (int idx = 0; idx < fuse_times; ++idx) {
      ir_sch.Fuse(block_name, {axes[index + 1], axes[index + 1] + 1});
    }
    LoopOrderAssignReduce(ir_sch, block_name, first_axes, target, true);
    // fuse axis before reduce to bind blockidx.
    for (int idx = 0; idx < (inshape.size() - axes.size()) - 1; ++idx) {
      ir_sch.Fuse(block_name, {0, 1});
    }
  }
}

inline bool CanbeInline(Node* node,
                        const std::vector<Node*> consumers,
                        const Node* reducer,
                        const Node* laster,
                        const GroupPtr& group,
                        const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  if (group->output_nodes.count(node)) {
    return false;
  }
  if (IsConstOp(node)) {
    return true;
  }
  if (reducer) {
    auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
    if (op_pattern_dict[node->op()] == framework::kReduction) {
      return false;
    }

    auto node_shape  = GetOutputShape(node, shape_dict);
    auto input_shape = GetInputShape(reducer, shape_dict);

    if (std::accumulate(node_shape.begin(), node_shape.end(), 1, std::multiplies<int>()) !=
        std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>())) {
      return true;
    }

    if (consumers.size() == 1) {
      return true;
    }

    return false;
  } else {
    auto node_shape = GetOutputShape(node, shape_dict);
    auto last_shape = GetOutputShape(laster, shape_dict);
    if (std::accumulate(node_shape.begin(), node_shape.end(), 1, std::multiplies<int>()) !=
        std::accumulate(last_shape.begin(), last_shape.end(), 1, std::multiplies<int>())) {
      return true;
    }

    if (consumers.size() == 1) {
      return true;
    }

    return false;
  }
}

inline Node* GetMasterToComputeAt(Node* node,
                                  const std::unordered_set<Node*>& nodes_inline,
                                  const std::unordered_set<Node*>& node_set) {
  std::queue<Node*> candidates;
  auto consumers = GetConsumers(node, node_set);
  for (auto consumer : consumers) {
    if (nodes_inline.count(consumer)) {
      candidates.push(consumer);
      continue;
    } else {
      return consumer;
    }
  }

  std::unordered_set<Node*> visited;
  while (!candidates.empty()) {
    auto candidate = candidates.front();
    candidates.pop();

    for (auto consumer : GetConsumers(candidate, node_set)) {
      if (visited.count(consumer)) {
        continue;
      }
      if (nodes_inline.count(consumer)) {
        candidates.push(consumer);
        visited.insert(consumer);
      } else {
        return candidate;
      }
    }
  }

  return nullptr;
}

inline void LoopAssignReduce(ir::IRSchedule& ir_sch,
                             const Node* node,
                             const Node* reducer,
                             const Target& target,
                             const std::unordered_map<std::string, ir::Tensor>& tensor_map,
                             const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  // if node is reducer, return.
  if (op_pattern_dict[node->op()] == framework::kReduction) {
    return;
  }
  auto node_data    = GetNodeData(node);
  auto reducer_data = GetNodeData(reducer);

  // flatten loops.
  auto loops = ir_sch.GetLoops(node_data->id());
  // do loop flatten.
  if (op_pattern_dict[node->op()] == framework::kElementWise) {
    ir_sch.FlattenLoops(loops, true);
  } else {
    ir_sch.FlattenLoops(loops, false);
  }

  // shape and axis.
  CHECK(shape_dict.count(reducer->inlinks_in_order()[0]->source()->id()));
  auto shape = shape_dict.at(reducer->inlinks_in_order()[0]->source()->id());
  auto axes  = absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
  if (axes.empty()) {
    for (int idx = 0; idx < shape.size(); idx++) {
      axes.push_back(idx);
    }
  }

  auto node_shape = shape_dict.at(node_data->id());
  // node output is same shape with reduce output.
  if (std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) !=
      std::accumulate(node_shape.begin(), node_shape.end(), 1, std::multiplies<int>())) {
    // split loop to assign master loop
    int extend = 1;
    std::vector<int> factors;
    loops       = ir_sch.GetLoops(node_data->id());
    auto rloops = ir_sch.GetLoops(reducer_data->id());

    for (auto& loop : rloops) {
      extend *= loop.As<ir::For>()->extent.as_int32();
      if (extend > loops.back().As<ir::For>()->extent.as_int32()) {
        break;
      }
      CHECK_LE(extend, loops.back().As<ir::For>()->extent.as_int32());
      factors.push_back(loop.As<ir::For>()->extent.as_int32());
    }

    ir_sch.Split(loops.back(), factors);
    loops = ir_sch.GetLoops(node_data->id());
    // copy loop info form rloops.
    for (int idx = 0; idx < std::min(rloops.size(), loops.size()); ++idx) {
      auto l0 = rloops[idx].As<ir::For>();
      auto l1 = loops[idx].As<ir::For>();
      l1->set_for_type(l0->for_type());
      l1->set_bind_info(l0->bind_info());
    }
    return;
  }

  // node output is same shape with reduce input.
  if (WithoutLastDimInReduce(shape, axes)) {
    // if using block shuffle
    if (tensor_map.count(reducer_data->id() + "_1")) {
      LoopAssignReduceWithoutLast(ir_sch, node_data->id(), shape, axes, target);
    } else {
      LoopOrderAssignReduce(ir_sch, node_data->id(), axes, target);
    }
  } else {
    if (tensor_map.count(reducer_data->id() + "_1")) {
      LoopAssignReduceWithLast(ir_sch, node_data->id(), shape, axes, target);
    } else if (tensor_map.count(reducer_data->id() + "_0")) {
    } else {
      LOG(FATAL) << "Error! Unkown Reduce Type!";
    }
  }
}

inline void MergeLoops(ir::Expr root, std::vector<ir::Expr>& src, std::vector<ir::Expr>& dst, int index) {
  CHECK_GT(src.size(), index);
  CHECK_GT(dst.size(), index);

  std::vector<ir::Var> src_vars;
  std::vector<ir::Expr> dst_vars;
  for (int idx = 0; idx <= index; ++idx) {
    src_vars.push_back(src[idx].As<ir::For>()->loop_var);
    dst_vars.push_back(ir::Expr(dst[idx].As<ir::For>()->loop_var));
  }

  auto src_body                  = src[index].As<ir::For>()->body;
  src[index].As<ir::For>()->body = ir::Expr();
  ReplaceExpr(&src_body, src_vars, dst_vars);
  dst[index].As<ir::For>()->body = ir::Block::Make({src_body, dst[index].As<ir::For>()->body});

  CHECK(root.As<ir::Block>());
  CHECK(root.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>());
  CHECK(root.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>());

  auto body =
      root.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body;
  CHECK(body.As<ir::Block>());
  auto block = body.As<ir::Block>();

  auto iter = std::find(block->stmts.begin(), block->stmts.end(), src[0]);
  CHECK(iter != block->stmts.end());
  block->stmts.erase(iter);
}

inline void LoopComputeAt(ir::IRSchedule& ir_sch,
                          Node* node,
                          const Node* master,
                          const GroupPtr& group,
                          const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  if (!master) return;

  auto node_data   = GetNodeData(node);
  auto master_data = GetNodeData(master);

  auto node_loops   = ir_sch.GetLoops(node_data->id());
  auto master_loops = ir_sch.GetLoops(master_data->id());

  if (!group->output_nodes.count(node)) {
    auto block = ir_sch.GetBlock(node_data->id());
    ir_sch.SetBuffer(block, "local", true);
  }

  int index             = std::min(node_loops.size(), master_loops.size()) - 1;
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  do {
    // if loop range is not equal.
    if (node_loops[index].As<ir::For>()->extent.as_int32() != master_loops[index].As<ir::For>()->extent.as_int32()) {
      continue;
    }

    std::string post = "";
    for (int idx = 0;; ++idx) {
      if (!tensor_map.count(node_data->id() + post)) {
        break;
      }

      auto tensor = tensor_map.find(node_data->id() + post)->second;
      if (!ir_sch.HasBlock(tensor->name)) {
        break;
      }
      auto src_loops = ir_sch.GetLoops(tensor->name);
      auto dst_loops = ir_sch.GetLoops(master_data->id());
      // ir_sch.SimpleComputeAt(src_loops[index], dst_loops[index]);
      MergeLoops(ir_sch.GetModule().GetExprs().at(0), src_loops, dst_loops, index);
      post = "_" + std::to_string(idx);
    }

    break;
  } while (--index);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

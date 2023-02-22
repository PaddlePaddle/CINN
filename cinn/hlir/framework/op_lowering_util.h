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

inline std::vector<Node*> GetConsumers(Node* node) {
  std::vector<Node*> consumers;
  auto node_data = GetNodeData(node);
  for (auto& link : node_data->outlinks_in_order(true)) {
    auto consumer = link->sink()->safe_as<Node>();
    CHECK(consumer);
    consumers.push_back(consumer);
  }
  return consumers;
}

inline std::vector<Node*> GetConsumers(Node* node, std::unordered_set<Node*> node_set) {
  std::vector<Node*> consumers;
  auto node_data = GetNodeData(node);
  for (auto& link : node_data->outlinks_in_order(true)) {
    auto consumer = link->sink()->safe_as<Node>();
    CHECK(consumer);
    if (node_set.count(consumer)) {
      consumers.push_back(consumer);
    }
  }
  return consumers;
}

inline std::vector<Node*> GetProducers(Node* node) {
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
      nodes_in_order.push(node);
      node_set.erase(node);
    }
  }

  return nodes_in_order;
}

inline Node* FindReducer(std::vector<Node*> node_in_order) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  for (auto iter = node_in_order.rbegin(); iter = node_in_order.rend(); ++iter) {
    if (op_pattern_dict[(*iter)->op()] == framework::kReduction) {
      return *iter;
    }
  }

  return nullptr;
}

inline void WithoutLastDimInReduce(const std::vector<int>& shape, const std::vector<int>& axes) {
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
                                        const common::Target& target,
                                        const std::vector<int>& axes) {
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
                                     const common::Target& target,
                                     const std::vector<int>& axes) {
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

inline bool CanbeInline(const Node* node,
                        const std::vector<Node*> consumers,
                        const Node* reducer,
                        const Node* laster,
                        const GroupPtr& group,
                        const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  if (reducer) {
    // if (op_pattern_dict[node->op()] == framework::kReduction) {
    //  return false;
    // }
    if (group->master_nodes.count(node)) {
      return false;
    }

    auto node_shape  = GetOutputShape(node);
    auto input_shape = GetInputShape(reducer);

    if (std::accumulate(node_shape.begin(), node_shape.end(), 1, std::multiplies<int>()) !=
        std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>())) {
      return true;
    }

    if (consumers.size() == 1) {
      return true;
    }

    return false;
  } else {
    auto node_shape = GetOutputShape(node);
    auto last_shape = GetOutputShape(laster);
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

inline Node* GetMasterToComputeAt(const Node* node,
                                  std::unordered_set<Node*> nodes_inline,
                                  std::unordered_set<Node*> node_set) {
  std::queue<Node*> candidates;
  for (auto consumer : GetConsumers(node, node_set)) {
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
                             const Node* master,
                             const Node* reducer,
                             const std::unordered_map<std::string, ir::Tensor>& tensor_map,
                             const absl::flat_hash_map<std::string, shape_t>& shape_dict) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  // if node is reducer, return.
  if (op_pattern_dict[node->op()] == framework::kReduction || !reducer) {
    return;
  }
  auto node_data    = GetNodeData(node);
  auto master_data  = GetNodeData(master);
  auto reducer_data = GetNodeData(reducer);

  // get node loops
  auto loops = ir_sch.GetLoops(node_data->id());
  // do loop flatten.
  if (op_pattern_dict[master->op()] == framework::kElementWise) {
    ir_sch.FlattenLoops(loops, true);
  } else {
    ir_sch.FlattenLoops(loops, false);
  }

  CHECK(shape_dict.count(reducer->inlinks_in_order()[0]->source()->id()));
  auto shape = shape_dict.at(reducer->inlinks_in_order()[0]->source()->id());
  auto axes  = absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
  if (axes.empty()) {
    for (int idx = 0; idx < shape.size(); idx++) {
      axes.push_back(idx);
    }
  }

  auto node_shape = this->shape_dict_.at(node_data->id());
  // node output is same shape with reduce output.
  if (std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) !=
      std::accumulate(node_shape.begin(), node_shape.end(), 1, std::multiplies<int>())) {
    // split loop to assign master loop
    std::vector<int> factors;
    auto mloops = ir_sch.GetLoops(master_tensor->name);
    for (auto& loop : mloops) {
      factors.push_back(loop.As<ir::For>()->extent.as_int32());
    }
    loops = ir_sch.GetLoops(node_tensor->name);
    ir_sch.Split(loops.back(), factors);
    return;
  }
  // node output is same shape with reduce input.
  if (WithoutLastDimInReduce(shape, axes)) {
    // if using block shuffle
    if (tensor_map.count(reducer_data->id() + "_1")) {
      LoopAssignReduceWithoutLast(ir_sch, node_data->id(), shape, axes);
    } else {
      LoopOrderAssignReduce(ir_sch, node_data->id(), shape, axes);
    }
  } else {
    if (tensor_map.count(reducer_data->id() + "_1")) {
      LoopAssignReduceWithLast(ir_sch, node_data->id(), shape, axes);
    } else if (tensor_map.count(reducer_data->id() + "_0")) {
    } else {
      LOG(FATAL) << "Error! Unkown Reduce Type!";
    }
  }
}

inline void LoopComputeAt(ir::IRSchedule& ir_sch,
                          const Node* node,
                          const Node* master,
                          const GroupPtr& group,
                          const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  if (!master) return;

  auto node_data   = GetNodeData(node);
  auto master_data = GetNodeData(master);

  auto node_loops   = ir_sch.GetLoops(node_data->id());
  auto master_loops = ir_sch.GetLoops(master_data->id());

  int index             = std::min(node_loops.size(), master_data.size()) - 1;
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  do {
    if (node_loops[index]->safe_as<ir::For>->extent.as_int32() ==
        master_loops[index]->safe_as<ir::For>->extent.as_int32()) {
      if (!group->master_nodes.count(node)) {
        ir_sch.SetBuffer(block, "local", true);
      }

      if (op_pattern_dict[node->op()] == framework::kReduction) {
        std::string post = "";
        for (int idx = 0;; ++idx) {
          if (!tensor_map.count(node_data->id() + post)) {
            break;
          }
          auto block = ir_sch.GetBlock(node_data->id() + post);
          ir_sch.SimpleComputeAt(block, node_loops[index]);
          post = "_" + std::to_string(idx);
        }
      } else if (op_pattern_dict[node->op()] == framework::kElementWise ||
                 op_pattern_dict[node->op()] == framework::kBroadcast ||
                 op_pattern_dict[node->op()] == framework::kInjective) {
        auto block = ir_sch.GetBlock(node_data->id());
        ir_sch.SimpleComputeAt(block, node_loops[index]);
        break;
      } else {
        LOG(FATAL) << "node type is unsupport now!";
      }
    }
  } while (--index);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

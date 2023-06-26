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

#include <string>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/ir_schedule_util.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/packed_func.h"

// Fusion Op lowering, there are four kinds of lowering function:
// Elementwise/Broadcast/Injective,Reduce,OutEWiseFusable,NonFusible.
// Elementwise/Broadcast/Injective Ops is with same shcedule.
// Reduce,OutEWiseFusable,NonFusible are using different schedule.

namespace cinn {
namespace hlir {
namespace framework {

using GroupPtr = std::shared_ptr<Graph::Group>;
using common::Target;

class OpLowerer;

typedef bool (OpLowerer::*ScheduleDetermineFunction)(Node*);

class OpLowerer {
 public:
  OpLowerer(const absl::flat_hash_map<std::string, Type>&,
            const absl::flat_hash_map<std::string, shape_t>&,
            const Target&);

  /**
   * @brief Lower a group to CINN IR
   * @param apply_op_schedule Whether to schedule at Op level.
   * @param apply_group_schedule Whether to schedule at group level.
   */
  std::vector<ir::LoweredFunc> Lower(GroupPtr& group, bool apply_op_schedule = true, bool apply_group_schedule = true);

 private:
  std::vector<ir::LoweredFunc> LowerGroup(GroupPtr& group,
                                          bool apply_op_schedule,
                                          bool apply_group_schedule,
                                          ScheduleDetermineFunction schedule_determine_func);

  std::vector<ir::LoweredFunc> LowerCustomCall(GroupPtr& group);

  std::vector<ir::LoweredFunc> PostProcess(ir::IRSchedule* ir_sch,
                                           const GroupPtr& group,
                                           const std::unordered_map<std::string, ir::Tensor>& tensor_map,
                                           std::vector<ir::Tensor>* group_func_arg_tensors,
                                           bool done_op_schedule);

  std::vector<ir::Expr> LowerOps(const std::vector<Node*>& nodes,
                                 std::vector<ir::Tensor>* group_func_arg_tensors,
                                 std::unordered_map<std::string, ir::Tensor>* tensor_map,
                                 bool apply_op_schedule,
                                 ScheduleDetermineFunction schedule_determine_func);

  std::vector<ir::LoweredFunc> DoOpLower(Node* node,
                                         std::shared_ptr<hlir::framework::OpImpl> op_impl,
                                         std::unordered_map<std::string, ir::Tensor>* tensor_map,
                                         std::vector<ir::Tensor>* op_func_arg_tensors);

  ir::Expr DoOpSchedule(std::shared_ptr<hlir::framework::OpImpl> op_impl,
                        const std::vector<ir::Tensor>& op_func_arg_tensors,
                        const std::vector<ir::LoweredFunc>& lowered_funcs);

  ir::Expr DoGroupSchedule(ir::IRSchedule& ir_sch,
                           const GroupPtr& group,
                           const std::unordered_map<std::string, ir::Tensor>& tensor_map);

  inline bool ReduceScheduleDetermineFunction(Node* node);
  inline bool ElementwiseScheduleDetermineFunction(Node* node);
  inline bool NonFusibleScheduleDetermineFunction(Node* node);

 private:
  Target target_;
  const absl::flat_hash_map<std::string, Type>& type_dict_;
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;

  // fucntion name prefix
  const std::string func_name_prefix = "fn_";
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

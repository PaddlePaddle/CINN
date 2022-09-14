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

#include <absl/container/flat_hash_map.h>

#include <memory>
#include <string>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op_lowering.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/lowered_func.h"

namespace cinn {
namespace auto_schedule {

class TuneTask {
 public:
  TuneTask() = default;
  TuneTask(hlir::framework::OpLowerer* op_lowerer) : op_lowerer_(op_lowerer) {}

  void SetOpLowerer(hlir::framework::OpLowerer* op_lowerer);
  // Set lowered_funcs and analyze output names.
  void SetLoweredFuncsAndAnalyzeOutput(const std::vector<ir::LoweredFunc>& lowered_funcs);
  // Extract bodies in lowered_funcs() and return
  std::vector<ir::Expr> GetLoweredFuncBodyExprs() const;
  // Set bodies in lowered_funcs() by exprs
  void SetLoweredFuncBodyExprs(const std::vector<ir::Expr>& exprs);
  // When you set OpLowerer and task_graph, lower the task graph to
  // un-optimized LoweredFunc and store in lowered_funcs().
  void TaskGraphToUnoptLoweredFunc();
  // Serialize this task as a string contains specific fields of it
  const std::string& SerializeToString(const absl::flat_hash_map<std::string, hlir::framework::shape_t>& shape_dict,
                                       const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict);

  // In CINN, we use std::vector<hlir::framework::Node*> to represent a fused
  // sub-graph (if an op won't be fused, it will be a vector with size=1). So
  // the task_graph_ consist of multiple "fused sub-graph" / "unfused op"
  std::vector<std::shared_ptr<hlir::framework::Graph::Group>> task_graph;
  // target of this task
  common::Target target;
  // stores the initial (un-optimized) LoweredFuncs
  std::vector<ir::LoweredFunc> lowered_funcs;
  // names of the output arguments of lowered_funcs_
  std::unordered_set<std::string> output_names;
  // serialized string of this task, it contain struct,shape,dtype informat
  // and can be further used to hash
  std::string serialized_key;

 private:
  // Not owned
  hlir::framework::OpLowerer* op_lowerer_;
};

}  // namespace auto_schedule
}  // namespace cinn

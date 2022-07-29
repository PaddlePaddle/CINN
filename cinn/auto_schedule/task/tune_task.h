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

#include <memory>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/lowered_func.h"

namespace cinn {
namespace auto_schedule {

class TuneTask {
 public:
  TuneTask() = default;
  TuneTask(hlir::framework::GraphCompiler* compiler) : graph_compiler_(compiler) {}

  void SetGraphCompiler(hlir::framework::GraphCompiler* compiler);
  // Set lowered_funcs and analyze output names.
  void SetLoweredFuncsAndAnalyzeOutput(const std::vector<ir::LoweredFunc>& lowered_funcs);
  // Extract bodies in lowered_funcs() and return
  std::vector<ir::Expr> GetLoweredFuncBodyExprs() const;
  // Set bodies in lowered_funcs() by exprs
  void SetLoweredFuncBodyExprs(const std::vector<ir::Expr>& exprs);
  // When you set GraphCompiler and task_graph, lower the task graph to
  // un-optimized LoweredFunc and store in lowered_funcs().
  void TaskGraphToUnoptLoweredFunc();

  // In CINN, we use std::vector<hlir::framework::Node*> to represent a fused
  // sub-graph (if an op won't be fused, it will be a vector with size=1). So
  // the task_graph_ consist of multiple "fused sub-graph" / "unfused op"
  std::vector<std::vector<hlir::framework::Node*>> task_graph;
  // target of this task
  common::Target target;
  // stores the initial (un-optimized) LoweredFuncs
  std::vector<ir::LoweredFunc> lowered_funcs;
  // names of the output arguments of lowered_funcs_
  std::unordered_set<std::string> output_names;

 private:
  // Not owned
  hlir::framework::GraphCompiler* graph_compiler_;
};

}  // namespace auto_schedule
}  // namespace cinn

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

#include "cinn/auto_schedule/task/tune_context.h"
#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class TuneTask {
 public:
  TuneTask() = default;

  TuneTask(hlir::framework::GraphCompiler* compiler) : graph_compiler_(compiler) {}

  std::vector<std::vector<hlir::framework::Node*>>& task_graph() { return task_graph_; }

  TuneContext& tune_context() { return tune_context_; }

  const TuneContext& tune_context() const { return tune_context_; }

  void SetGraphCompiler(hlir::framework::GraphCompiler* compiler);

  void TaskGraphToUnoptLoweredFunc();

 private:
  // In CINN, we use std::vector<hlir::framework::Node*> to represent a fused
  // sub-graph (if an op won't be fused, it will be a vector with size=1). So
  // the task_graph_ consist of multiple "fused sub-graph" / "unfused op"
  std::vector<std::vector<hlir::framework::Node*>> task_graph_;
  // Context of a tune task
  TuneContext tune_context_;
  // Not owned
  hlir::framework::GraphCompiler* graph_compiler_;
};

}  // namespace auto_schedule
}  // namespace cinn

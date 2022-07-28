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

#include "cinn/auto_schedule/task/tune_task.h"

#include <glog/logging.h>

#include <vector>

#include "cinn/auto_schedule/analysis/analyze_ir.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/lowered_func.h"

namespace cinn {
namespace auto_schedule {

void TuneTask::SetGraphCompiler(hlir::framework::GraphCompiler* compiler) { graph_compiler_ = compiler; }

const common::Target& TuneTask::GetTarget() const { return target_; };

void TuneTask::SetTarget(const common::Target& target) { target_ = target; }

std::vector<ir::Expr> TuneTask::GetLoweredFuncBodyExprs() const {
  std::vector<ir::Expr> result;
  for (const ir::LoweredFunc& func : lowered_funcs_) {
    result.push_back(func->body);
  }
  return result;
}

void TuneTask::SetLoweredFuncBodyExprs(const std::vector<ir::Expr>& exprs) {
  size_t exprs_size = exprs.size();
  CHECK_EQ(exprs_size, lowered_funcs_.size())
      << "SetLoweredFuncBodyExprs must have same number of Expr(s) and LoweredFunc(s)";
  for (size_t i = 0; i < exprs_size; ++i) {
    lowered_funcs_[i]->body = exprs[i];
  }
}

void TuneTask::SetLoweredFuncsAndAnalyzeOutput(const std::vector<ir::LoweredFunc>& lowered_funcs) {
  this->lowered_funcs_ = lowered_funcs;
  this->output_names_  = GetOutputNamesFromLoweredFunc(this->lowered_funcs_);
}

void TuneTask::TaskGraphToUnoptLoweredFunc() {
  CHECK(graph_compiler_ != nullptr) << "graph_compiler_ must be set before processing graph";
  // TODO(zhhsplendid): current a task only contains one Op or one Fused Op,
  // so we can take only first std::vector<ir::LoweredFunc>. Support the
  // lowered_funcs to be std::vector<std::vector<ir::LoweredFunc>>
  // in the future.
  SetLoweredFuncsAndAnalyzeOutput(graph_compiler_->FusedGraphToLoweredFunc(task_graph_)[0]);
}

}  // namespace auto_schedule
}  // namespace cinn

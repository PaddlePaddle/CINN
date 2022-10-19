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

#include <iostream>
#include <vector>

#include "cinn/auto_schedule/analysis/analyze_ir.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op_lowering.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

void TuneTask::SetOpLowerer(hlir::framework::OpLowerer* op_lowerer) { op_lowerer_ = op_lowerer; }

std::vector<ir::Expr> TuneTask::GetLoweredFuncBodyExprs() const {
  std::vector<ir::Expr> result;
  for (const ir::LoweredFunc& func : lowered_funcs) {
    result.push_back(func->body);
  }
  return result;
}

void TuneTask::SetLoweredFuncBodyExprs(const std::vector<ir::Expr>& exprs) {
  size_t exprs_size = exprs.size();
  CHECK_EQ(exprs_size, lowered_funcs.size())
      << "SetLoweredFuncBodyExprs must have same number of Expr(s) and LoweredFunc(s)";
  for (size_t i = 0; i < exprs_size; ++i) {
    lowered_funcs[i]->body = exprs[i];
  }
}

void TuneTask::SetLoweredFuncsAndAnalyzeOutput(const std::vector<ir::LoweredFunc>& lowered_funcs) {
  this->lowered_funcs = lowered_funcs;
  this->output_names  = GetOutputNamesFromLoweredFunc(this->lowered_funcs);
}

void TuneTask::TaskGraphToUnoptLoweredFunc() {
  CHECK(op_lowerer_ != nullptr) << "op_lowerer_ must be set before processing graph";

  // TODO(zhhsplendid): current a task only contains one Op or one Fused Op,
  // so we can take only first group to lower to std::vector<ir::LoweredFunc>.
  // Support the lowered_funcs to be std::vector<std::vector<ir::LoweredFunc>>
  // in the future.
  SetLoweredFuncsAndAnalyzeOutput(op_lowerer_->LowerWithoutSchedule(task_graph[0]));
}

const std::string& TuneTask::SerializeToString(
    const absl::flat_hash_map<std::string, hlir::framework::shape_t>& shape_dict,
    const absl::flat_hash_map<std::string, cinn::common::Type>& dtype_dict) {
  std::stringstream ss;
  ss << target << "\n\n";  // print target

  // local function to print dtype,shape of out/in variables of the specified node
  auto print_node_links_fn = [&](const std::vector<common::Shared<common::GraphEdge>>& links, bool is_input) {
    int printed_num = 0;
    for (auto&& edge : links) {
      const auto* var_node = is_input ? edge->source()->safe_as<hlir::framework::NodeData>()
                                      : edge->sink()->safe_as<hlir::framework::NodeData>();
      CHECK(var_node) << "var node invalid";
      auto sit = shape_dict.find(var_node->id());
      CHECK(sit != shape_dict.end()) << "can't find shape of variable:" << var_node->id();
      auto dit = dtype_dict.find(var_node->id());
      CHECK(dit != dtype_dict.end()) << "can't find dtype of variable:" << var_node->id();
      if (printed_num > 0) {
        ss << ", ";
      }
      ++printed_num;
      // TODO(CtfGo): CINN uses the names of input/output NodeData as arguments of the LoweredFunc,
      // so it will result in different LoweredFuncs for two Nodes even though they represents the same operator.
      // Here we add `var_node->id()` into the serialized_key to distinguish them, otherwise AutoTuner will
      // get wrong TuningRecords in quering cached results from database.  In the future, we should remove name-releated
      // limit in Lower process, to avoid duplicate tuning tasks with same operators.
      ss << var_node->id() << "->" << cinn::common::Type2Str(dit->second) << "[" + utils::Join(sit->second, ",") << "]";
    }
  };

  // print each group of the task_graph
  for (auto p = 0; p < task_graph.size(); ++p) {
    const std::vector<hlir::framework::Node*>& group = task_graph.at(p)->CollectNodes();
    ss << "Group " << p << " {\n";
    for (auto i = 0; i < group.size(); ++i) {
      const hlir::framework::Node* node = group.at(i);
      ss << "  (";
      print_node_links_fn(node->outlinks_in_order(), false);
      ss << ") = " << node->op()->name << "(";
      print_node_links_fn(node->inlinks_in_order(), true);
      ss << ")\n";
    }
    ss << "}\n";
  }

  serialized_key = ss.str();
  return serialized_key;
}

}  // namespace auto_schedule
}  // namespace cinn

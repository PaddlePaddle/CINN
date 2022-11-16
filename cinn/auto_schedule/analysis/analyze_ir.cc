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

#include "cinn/auto_schedule/analysis/analyze_ir.h"

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <unordered_set>

#include "cinn/ir/buffer.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/ir/tensor.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace auto_schedule {

std::vector<ir::Var> IndicesToVars(const std::vector<ir::Expr>& indices) {
  std::vector<ir::Var> result;
  for (const ir::Expr& e : indices) {
    // Whether we have to convert other types, like const numbers to Var?
    if (e.As<ir::_Var_>() != nullptr) {
      ir::Expr copy_e    = optim::IRCopy(e);
      ir::_Var_* var_ref = copy_e.As<ir::_Var_>();
      result.emplace_back(ir::Var(var_ref));
    }
  }
  return result;
}

void AnalyzeScheduleBlockReadWriteBuffer(ir::ScheduleBlock* sche_block) {
  if (!sche_block->read_buffers.empty() || !sche_block->write_buffers.empty()) {
    return;
  }

  ir::CollectIRNodesWithoutTensor(sche_block->body, [&](const Expr* x) {
    const ir::Load* load_expr = x->As<ir::Load>();
    if (load_expr != nullptr) {
      const ir::Tensor t = load_expr->tensor.as_tensor_ref();
      sche_block->read_buffers.emplace_back(ir::BufferRange(t->buffer, IndicesToVars(load_expr->indices)));
      return false;
    }
    const ir::Store* store_expr = x->As<ir::Store>();
    if (store_expr != nullptr) {
      const ir::Tensor t = store_expr->tensor.as_tensor_ref();
      sche_block->write_buffers.emplace_back(ir::BufferRange(t->buffer, IndicesToVars(store_expr->indices)));
      return false;
    }
    return false;
  });
}

bool ContainsNodeType(ir::Expr expr, const std::unordered_set<ir::IrNodeTy>& node_types) {
  std::set<ir::Expr> collection = ir::CollectIRNodesWithoutTensor(
      expr, [&](const Expr* x) { return node_types.find(x->node_type()) != node_types.end(); });
  return !collection.empty();
}

std::unordered_set<std::string> GetOutputNamesFromLoweredFunc(const std::vector<ir::LoweredFunc>& lowered_funcs) {
  std::unordered_set<std::string> result;
  for (const ir::LoweredFunc& func : lowered_funcs) {
    for (const ir::Argument& arg : func->args) {
      if (arg.is_output()) {
        result.insert(arg.name());
      }
    }
  }
  return result;
}

}  // namespace auto_schedule
}  // namespace cinn

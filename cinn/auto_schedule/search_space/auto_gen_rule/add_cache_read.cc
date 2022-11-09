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

#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_read.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/analysis/analyze_ir.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/common/target.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace auto_schedule {

RuleApplyType AddCacheRead::Init(ir::IRSchedule* ir_schedule) {
  ir_schedule_        = ir_schedule;
  auto block_realizes = ir_schedule_->GetAllBlocks();
  applicable_schedule_blocks_.clear();
  num_applicable_ = 0;
  for (size_t i = 0; i < block_realizes.size(); ++i) {
    ir::ScheduleBlockRealize* sch_block_realize = block_realizes[i].As<ir::ScheduleBlockRealize>();
    // AnalyzeScheduleBlockReadWriteBuffer(sch_block_realize->schedule_block.As<ir::ScheduleBlock>());
    if (MeetCondition(*sch_block_realize)) {
      ++num_applicable_;
      applicable_schedule_blocks_.push_back(block_realizes[i]);
    }
  }
  VLOG(6) << "Collect applicable_schedule_blocks_:" << num_applicable_;

  cache_memory_type_ = kMemoryTypes.at(target_->arch);

  return num_applicable_ > 0 ? RuleApplyType::kApplyAndSkipAllRules : RuleApplyType::kCannotApply;
}

void AddCacheRead::Apply(int index) {
  ir::Expr sch_block_expr                     = applicable_schedule_blocks_[index];
  ir::ScheduleBlockRealize* sch_block_realize = sch_block_expr.As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sch_block                = sch_block_realize->schedule_block.As<ir::ScheduleBlock>();
  std::string block_name                      = sch_block->name;

  std::vector<std::string> read_tensor_strs;
  ir::CollectIRNodesWithoutTensor(sch_block->body, [&](const Expr* x) {
    if (x->As<ir::Load>()) read_tensor_strs.push_back(x->As<ir::Load>()->tensor.as_tensor()->name);
    return x->As<ir::Load>() != nullptr;
  });
  for (int i = 0; i < read_tensor_strs.size(); ++i) {
    VLOG(6) << "read tensors[" << i << "]: " << read_tensor_strs[i];
  }

  std::unordered_set<std::string> write_tensor_strs;
  ir::CollectIRNodesWithoutTensor(sch_block->body, [&](const Expr* x) {
    if (x->As<ir::Store>()) write_tensor_strs.insert(x->As<ir::Store>()->tensor.as_tensor()->name);
    return x->As<ir::Store>() != nullptr;
  });
  for (const auto& write_tensor_str : write_tensor_strs) {
    VLOG(6) << "write tensors: " << write_tensor_str;
  }

  std::vector<int> read_buffer_indexes;
  for (int i = 0; i < read_tensor_strs.size(); ++i) {
    if (write_tensor_strs.count(read_tensor_strs[i]) == 0) {
      read_buffer_indexes.push_back(i);
    }
  }
  for (int idx : read_buffer_indexes) {
    VLOG(6) << "only read tensors: " << read_tensor_strs[idx];
  }

  for (int read_buffer_index : read_buffer_indexes) {
    ir::Expr cache_block = ir_schedule_->CacheRead(sch_block_expr, read_buffer_index, cache_memory_type_);
    VLOG(6) << "cache block: " << cache_block;
    sch_block_expr       = ir_schedule_->GetBlock(block_name);
    ir::Expr target_loop = GetTargetLoop(sch_block_expr);
    ir_schedule_->ComputeAt(cache_block, target_loop);
  }
}

bool AddCacheRead::MeetCondition(const ir::ScheduleBlockRealize& sche_block_realize) const {
  const ir::ScheduleBlock* sch_block = sche_block_realize.schedule_block.As<ir::ScheduleBlock>();
  if (sch_block->read_buffers.empty() || sch_block->write_buffers.size() != 1) {
    return false;
  }
  return true;
}

ir::Expr AddCacheRead::GetTargetLoop(const ir::Expr& block_expr) const {
  std::vector<Expr> for_exprs = ir_schedule_->GetLoops(block_expr);
  for (auto& for_expr : for_exprs) {
    ir::Var for_node_var          = for_expr.As<ir::For>()->loop_var;
    std::string for_loop_var_name = for_node_var->name;
    if (for_loop_var_name.substr(0, 11) == "reduce_axis") {
      VLOG(6) << "get target loop: " << for_expr;
      return for_expr;
    }
  }

  LOG(FATAL) << "Cannot find target loop.";
  return for_exprs[0];
}

const std::unordered_map<common::Target::Arch, std::string> AddCacheRead::kMemoryTypes{
    {common::Target::Arch::X86, "local"}, {common::Target::Arch::NVGPU, "shared"}};

}  // namespace auto_schedule
}  // namespace cinn
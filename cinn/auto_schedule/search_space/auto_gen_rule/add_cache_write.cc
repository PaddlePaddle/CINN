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

#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_write.h"

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
#include "cinn/ir/ir_schedule_util.h"
#include "cinn/ir/tensor.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace auto_schedule {

RuleApplyType AddCacheWrite::Init(ir::IRSchedule* ir_schedule) {
  ir_schedule_        = ir_schedule;
  auto block_realizes = ir_schedule_->GetAllBlocks();
  applicable_schedule_blocks_.clear();
  num_applicable_ = 0;
  for (size_t i = 0; i < block_realizes.size(); ++i) {
    ir::ScheduleBlockRealize* sch_block_realize = block_realizes[i].As<ir::ScheduleBlockRealize>();
    // Prepare the read/write buffer information of the block,
    // which will be used to analyze which buffers can be cached.
    AnalyzeScheduleBlockReadWriteBuffer(sch_block_realize->schedule_block.As<ir::ScheduleBlock>());
    // Use function MeetCondition() to filter inapplicable blocks,
    // only save the applicable blocks, and the index will be used for subsequent access.
    if (MeetCondition(ir_schedule, block_realizes[i])) {
      ++num_applicable_;
      applicable_schedule_blocks_.push_back(block_realizes[i]);
    }
  }
  VLOG(6) << "Collect applicable_schedule_blocks_:" << num_applicable_;

  if (num_applicable_ > 0) {
    if (*target_ == common::DefaultNVGPUTarget()) return RuleApplyType::kApplyAndPruneOtherRules;
    if (*target_ == common::DefaultHostTarget()) return RuleApplyType::kApply;
  }

  return RuleApplyType::kCannotApply;
}

ir::Expr AddCacheWrite::GetFirstSpatialLoopOutofOutermostReduce(ir::IRSchedule* ir_schedule,
                                                                const ir::Expr& block) const {
  VLOG(6) << block;
  std::vector<ir::Expr> for_exprs = ir_schedule->GetLoops(block);
  ir::Expr spatial_loop(nullptr);
  for (auto& for_expr : for_exprs) {
    ir::Var for_node_var          = for_expr.As<ir::For>()->loop_var;
    std::string for_loop_var_name = for_node_var->name;
    if (for_loop_var_name.substr(0, 6) == "reduce") {
      return spatial_loop;
    }
    spatial_loop = for_expr;
  }

  LOG(FATAL) << "Cannot find target loop.";
  return for_exprs[0];
}

void AddCacheWrite::Apply(int index) {
  ir::Expr sch_block_expr = applicable_schedule_blocks_[index];

  // Schedule
  Apply(ir_schedule_, sch_block_expr);
}

bool AddCacheWrite::MeetCondition(ir::IRSchedule* ir_schedule, const ir::Expr& block_expr) const {
  const ir::ScheduleBlockRealize* sch_block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  const ir::ScheduleBlock* sch_block                = sch_block_realize->schedule_block.As<ir::ScheduleBlock>();
  std::vector<ir::Expr> for_exprs                   = ir_schedule->GetLoops(block_expr);
  ir::Expr spatial_loop(nullptr);
  for (auto& for_expr : for_exprs) {
    ir::Var for_node_var          = for_expr.As<ir::For>()->loop_var;
    std::string for_loop_var_name = for_node_var->name;
    if (for_loop_var_name.substr(0, 6) == "reduce") {
      return NeedsMultiLevelTiling(*sch_block_realize);
    }
  }

  return false;
}

RuleApplyType AddCacheWrite::AnalyseApplyType(SearchState state, const std::string& block_name) const {
  Expr block_expr     = state->ir_schedule.GetBlock(block_name);
  auto* block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  CHECK(block_realize) << "stmt is not a ScheduleBlockRealize:" << block_expr;
  // Prepare the read/write buffer information of the block,
  // which will be used to analyze which buffers can be cached.
  AnalyzeScheduleBlockReadWriteBuffer(block_realize->schedule_block.As<ir::ScheduleBlock>());
  if (MeetCondition(&(state->ir_schedule), block_realize)) {
    if (*target_ == common::DefaultNVGPUTarget()) return RuleApplyType::kApplyAndPruneOtherRules;
    if (*target_ == common::DefaultHostTarget()) return RuleApplyType::kApply;
  }

  return RuleApplyType::kCannotApply;
}

std::vector<SearchState> AddCacheWrite::ApplyOnBlock(SearchState state, const std::string& block_name) {
  SearchState new_state   = state.Copy();
  ir::IRSchedule* ir_sch  = &new_state->ir_schedule;
  ir::Expr sch_block_expr = ir_sch->GetBlock(block_name);
  Apply(ir_sch, sch_block_expr);

  return {new_state};
}

void AddCacheWrite::Apply(ir::IRSchedule* ir_schedule, ir::Expr& block_expr) {
  ir::Expr cache_block = ir_schedule->CacheWrite(block_expr, 0, cache_memory_type_);
  VLOG(6) << "cache block: " << cache_block;
  ir::Expr target_loop = GetFirstSpatialLoopOutofOutermostReduce(ir_schedule, cache_block);
  VLOG(6) << "target_loop: " << target_loop;
  const std::string& original_block_name =
      block_expr.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
  ir_schedule->ReverseComputeAt(ir_schedule->GetBlock(original_block_name), target_loop);

  target_loop                              = GetFirstSpatialLoopOutofOutermostReduce(ir_schedule, cache_block);
  const std::string reduce_init_block_name = original_block_name + "__reduce_init";
  if (ir_schedule->HasBlock(reduce_init_block_name)) {
    ir_schedule->SimpleComputeAt(ir_schedule->GetBlock(reduce_init_block_name), target_loop);
  }
}

const std::unordered_map<common::Target::Arch, std::string> AddCacheWrite::kMemoryTypes{
    {common::Target::Arch::X86, "local"}, {common::Target::Arch::NVGPU, "local"}};

}  // namespace auto_schedule
}  // namespace cinn

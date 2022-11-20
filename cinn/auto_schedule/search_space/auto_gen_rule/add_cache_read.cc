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
    // Prepare the read/write buffer information of the block,
    // which will be used to analyze which buffers can be cached.
    AnalyzeScheduleBlockReadWriteBuffer(sch_block_realize->schedule_block.As<ir::ScheduleBlock>());
    // Use function MeetCondition() to filter inapplicable blocks,
    // only save the applicable blocks, and the index will be used for subsequent access.
    if (MeetCondition(block_realizes[i])) {
      ++num_applicable_;
      applicable_schedule_blocks_.push_back(block_realizes[i]);
    }
  }
  VLOG(6) << "Collect applicable_schedule_blocks_:" << num_applicable_;

  // Select a cache memory type
  cache_memory_type_ = kMemoryTypes.at(target_->arch);

  return num_applicable_ > 0 ? RuleApplyType::kApplyAndSkipThisRule : RuleApplyType::kCannotApply;
}

void AddCacheRead::Apply(int index) {
  ir::Expr sch_block_expr                     = applicable_schedule_blocks_[index];
  ir::ScheduleBlockRealize* sch_block_realize = sch_block_expr.As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sch_block                = sch_block_realize->schedule_block.As<ir::ScheduleBlock>();
  std::string block_name                      = sch_block->name;

  // Analyze which buffers can be cached
  std::vector<int> read_buffer_indexes;
  for (int i = 0; i < sch_block->read_buffers.size(); ++i) {
    bool is_read_write = false;
    for (int j = 0; j < sch_block->write_buffers.size(); ++j) {
      if (sch_block->read_buffers[i] == sch_block->write_buffers[j]) {
        is_read_write = true;
        break;
      }
    }
    if (!is_read_write) {
      read_buffer_indexes.push_back(i);
    }
  }

  // Schedule
  for (int read_buffer_index : read_buffer_indexes) {
    ir::Expr cache_block = ir_schedule_->CacheRead(sch_block_expr, read_buffer_index, cache_memory_type_);
    VLOG(6) << "cache block: " << cache_block;
    // The original block expr is invalid after the CacheRead schedule,
    // so we reacquire the block expr after the schedule according to the block name
    sch_block_expr       = ir_schedule_->GetBlock(block_name);
    ir::Expr target_loop = GetTargetLoop(sch_block_expr);
    ir_schedule_->ComputeAt(cache_block, target_loop);
  }
}

bool AddCacheRead::MeetCondition(const ir::Expr& block_expr) const {
  const ir::ScheduleBlockRealize* sch_block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  const ir::ScheduleBlock* sch_block                = sch_block_realize->schedule_block.As<ir::ScheduleBlock>();

  if (!NeedsMultiLevelTiling(*sch_block_realize)) return false;

  // check cross thread reduce axis
  for (const ir::Expr& for_expr : ir_schedule_->GetLoops(block_expr)) {
    const ir::For* for_node = for_expr.As<ir::For>();
    if (for_node->is_gpu_thread_binded() && for_node->loop_var->is_reduce_axis) {
      return false;
    }
  }

  return true;
}

ir::Expr AddCacheRead::GetTargetLoop(const ir::Expr& block_expr) const {
  // Get the out most reduce axis
  std::vector<ir::Expr> for_exprs = ir_schedule_->GetLoops(block_expr);
  for (auto& for_expr : for_exprs) {
    ir::Var for_node_var          = for_expr.As<ir::For>()->loop_var;
    std::string for_loop_var_name = for_node_var->name;
    if (for_loop_var_name.substr(0, 6) == "reduce") {
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

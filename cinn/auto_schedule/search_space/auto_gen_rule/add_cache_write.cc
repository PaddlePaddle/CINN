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
    AnalyzeScheduleBlockReadWriteBuffer(sch_block_realize->schedule_block.As<ir::ScheduleBlock>());
    if (MeetCondition(block_realizes[i])) {
      ++num_applicable_;
      applicable_schedule_blocks_.push_back(block_realizes[i]);
    }
  }
  VLOG(6) << "Collect applicable_schedule_blocks_:" << num_applicable_;

  cache_memory_type_ = kMemoryTypes.at(target_->arch);

  if (num_applicable_ > 0) {
    if (*target_ == common::DefaultNVGPUTarget()) return RuleApplyType::kApplyAndSkipAllRules;
    if (*target_ == common::DefaultHostTarget()) return RuleApplyType::kApplyAndSkipThisRule;
  }

  return RuleApplyType::kCannotApply;
}

void AddCacheWrite::Apply(int index) {
  ir::Expr sch_block_expr = applicable_schedule_blocks_[index];

  // Do schedule
  ir::Expr cache_block = ir_schedule_->CacheWrite(sch_block_expr, 0, cache_memory_type_);
  VLOG(6) << "cache block: " << cache_block;
}

// TODO: Merge this function and the same function in MultiLevelTiling rule
bool NeedMultiLevelTiling(const ir::ScheduleBlockRealize& sch_block_realize) {
  const ir::ScheduleBlock* sche_block = sch_block_realize.schedule_block.As<ir::ScheduleBlock>();
  const ir::Expr& write_buffer        = sche_block->write_buffers[0].As<ir::_BufferRange_>()->buffer;

  // Enumerate each read region, get the number of schedule block iter vars
  // which  are not used to index the read region
  int total_unused_iter_vars = 0;

  for (const ir::Expr& read_buffer_expr : sche_block->read_buffers) {
    const ir::_BufferRange_* read_buffer = read_buffer_expr.As<ir::_BufferRange_>();
    // Skip the reduction buffer
    if (read_buffer->buffer == write_buffer) {
      continue;
    }
    // Collect the vars in schedule block that are used to index the read region
    std::unordered_set<std::string> vars_index_read;
    for (const Var& range : read_buffer->ranges) {
      vars_index_read.insert(range->name);
    }
    // Check the block iter vars are not used to index the read region
    int n_unused_block_vars = 0;
    for (const ir::Var& block_iter_var : sche_block->iter_vars) {
      bool iter_var_in_read = false;
      for (const std::string& var : vars_index_read) {
        if (var == block_iter_var->name) {
          iter_var_in_read = true;
          break;
        }
      }
      if (!iter_var_in_read) {
        ++n_unused_block_vars;
      }
    }
    total_unused_iter_vars += n_unused_block_vars;
  }

  return total_unused_iter_vars >= 1;
}

bool AddCacheWrite::HasSingleElementwiseMatchedConsumer(const ir::Expr& block_expr) const {
  ir::Expr root_block_expr        = ir_schedule_->GetRootBlock(block_expr);
  std::vector<ir::Expr> consumers = ir::GetConsumers(block_expr, root_block_expr);
  VLOG(6) << "xb_debug consumers.size() = " << consumers.size();
  return false;
}

bool AddCacheWrite::MeetCondition(const ir::Expr& block_expr) const {
  const ir::ScheduleBlockRealize* sch_block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  const ir::ScheduleBlock* sch_block                = sch_block_realize->schedule_block.As<ir::ScheduleBlock>();

  if (sch_block->read_buffers.empty() || sch_block->write_buffers.size() != 1) {
    return false;
  }

  if (!NeedMultiLevelTiling(*sch_block_realize)) return false;
  if (HasSingleElementwiseMatchedConsumer(block_expr)) return false;

  return true;
}

const std::unordered_map<common::Target::Arch, std::string> AddCacheWrite::kMemoryTypes{
    {common::Target::Arch::X86, "local"}, {common::Target::Arch::NVGPU, "shared"}};

}  // namespace auto_schedule
}  // namespace cinn
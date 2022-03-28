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

#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"

#include <glog/logging.h>

#include <memory>
#include <string>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

bool MultiLevelTiling::MeetCondition(const ir::ScheduleBlockRealize& sche_block_realize) const {
  const ir::ScheduleBlock* sche_block = sche_block_realize.schedule_block.As<ir::ScheduleBlock>();
  if (sche_block->write_buffers.size() != 1 || sche_block->read_buffers.empty()) {
    return false;
  }
  const ir::Expr& write_buffer = sche_block->write_buffers[0]->buffer;

  // Enumerate each read region, get the number of schedule block iter vars
  // which  are not used to index the read region
  int total_unused_iter_vars = 0;

  for (const ir::BufferRange& read_buffer : sche_block->read_buffers) {
    // Skip the reduction buffer
    if (read_buffer->buffer == sche_block->write_buffers[0]->buffer) {
      continue;
    }
    // Collect the vars in schedule block that are used to index the read region
    std::unordered_set<const ir::Expr*> vars_index_read;
    for (const Var& range : read_buffer->ranges) {
      vars_index_read.insert(&(range->lower_bound));
      vars_index_read.insert(&(range->upper_bound));
    }
    // Check the block iter vars are not used to index the read region
    int n_unused_block_vars = 0;
    for (const ir::Var& block_iter_var : sche_block->iter_vars) {
      bool iter_var_in_read = false;
      for (const ir::Expr* expr : vars_index_read) {
        if ((*expr) == block_iter_var) {
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

bool MultiLevelTiling::Init(const ir::ModuleExpr& mod_expr) {
  ir_schedule_                             = std::make_unique<ir::IRSchedule>(mod_expr);
  std::vector<ir::Expr> all_block_realizes = ir_schedule_->GetAllBlocks();

  num_applicable_ = 0;
  for (size_t i = 0; i < all_block_realizes.size(); ++i) {
    const ir::ScheduleBlockRealize* sche_block_realize = all_block_realizes[i].As<ir::ScheduleBlockRealize>();
    if (MeetCondition(*sche_block_realize)) {
      ++num_applicable_;
      applicable_indices.push_back(i);
    }
  }

  return num_applicable_ > 0;
}

ir::ModuleExpr MultiLevelTiling::Apply(int index) { return ir::ModuleExpr(); }

std::string MultiLevelTiling::GetRuleName() const { return "MultiLevelTiling"; }

}  // namespace auto_schedule
}  // namespace cinn

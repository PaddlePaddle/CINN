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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/common/target.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace auto_schedule {

MultiLevelTiling::MultiLevelTiling(const common::Target& target) : AutoGenRule(target) {
  if (target == common::DefaultNVGPUTarget()) {
    bind_axis_   = {"blockIdx.x", "threadIdx.x"};
    tile_struct_ = "SSSRRSRS";
  } else {
    bind_axis_   = {};
    tile_struct_ = "SSRSRS";
  }

  for (int i = 0; i < tile_struct_.size(); ++i) {
    if (tile_struct_[i] == 'S') {
      s_indices_.push_back(i);
    } else if (tile_struct_[i] == 'R') {
      r_indices_.push_back(i);
    } else {
      CHECK(false) << "Illegal tiling structure string";
    }
  }
}

bool MultiLevelTiling::MeetCondition(const ir::ScheduleBlockRealize& sche_block_realize) const {
  const ir::ScheduleBlock* sche_block = sche_block_realize.schedule_block.As<ir::ScheduleBlock>();
  if (sche_block->write_buffers.size() != 1 || sche_block->read_buffers.empty()) {
    return false;
  }
  const ir::Expr& write_buffer = sche_block->write_buffers[0].As<ir::_BufferRange_>()->buffer;

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
        if ((*expr) == static_cast<Expr>(block_iter_var)) {
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

void MultiLevelTiling::AnalyzeScheduleBlockReadWriteBuffer(ir::ScheduleBlock* sche_block) const {
  if (!sche_block->read_buffers.empty() || !sche_block->write_buffers.empty()) {
    return;
  }

  std::set<ir::Expr> load_tensors = ir::CollectLoadTensors(sche_block->body, [&](const Expr* x) { return true; });
  for (const ir::Expr& e : load_tensors) {
    ir::Tensor t = e.as_tensor_ref();
    sche_block->read_buffers.emplace_back(ir::BufferRange(t->buffer, t->axis_with_reduce()));
  }

  std::set<ir::Expr> store_tensors = ir::CollectStoreTensors(sche_block->body, [&](const Expr* x) { return true; });
  for (const ir::Expr& e : store_tensors) {
    ir::Tensor t = e.as_tensor_ref();
    sche_block->write_buffers.emplace_back(ir::BufferRange(t->buffer, t->axis_with_reduce()));
  }

  auto buffer_range_cmp = [](const Expr& lhs, const Expr& rhs) {
    return lhs.As<ir::_BufferRange_>()->buffer.as_buffer_ref() < rhs.As<ir::_BufferRange_>()->buffer.as_buffer_ref();
  };
  sort(sche_block->read_buffers.begin(), sche_block->read_buffers.end(), buffer_range_cmp);
  sort(sche_block->write_buffers.begin(), sche_block->write_buffers.end(), buffer_range_cmp);
}

RuleApplyType MultiLevelTiling::Init(const ir::ModuleExpr& mod_expr) {
  ir_schedule_        = std::make_unique<ir::IRSchedule>(mod_expr);
  all_block_realizes_ = ir_schedule_->GetAllBlocks();
  applicable_indices_.clear();
  num_applicable_ = 0;
  for (size_t i = 0; i < all_block_realizes_.size(); ++i) {
    ir::ScheduleBlockRealize* sche_block_realize = all_block_realizes_[i].As<ir::ScheduleBlockRealize>();
    AnalyzeScheduleBlockReadWriteBuffer(sche_block_realize->schedule_block.As<ir::ScheduleBlock>());
    if (MeetCondition(*sche_block_realize)) {
      ++num_applicable_;
      applicable_indices_.push_back(i);
    }
  }

  return num_applicable_ > 0 ? RuleApplyType::kApplyAndSkipThisRule : RuleApplyType::kCannotApply;
}

ir::ModuleExpr MultiLevelTiling::Apply(int index) {
  CHECK(ir_schedule_ != nullptr) << "Run MultiLevelTiling::Apply without Init";
  CHECK(num_applicable_ > 0 && applicable_indices_.size() == num_applicable_)
      << "MultiLevelTiling::Apply pre-condition doesn't meet";
  CHECK(num_applicable_ > index)
      << "Invalid index for MultiLevelTiling::Apply, the index needs 0 <= index && index < NumberApplicable()";

  int apply_index                              = applicable_indices_[index];
  ir::ScheduleBlockRealize* sche_block_realize = all_block_realizes_[apply_index].As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sche_block                = sche_block_realize->schedule_block.As<ir::ScheduleBlock>();

  std::vector<Expr> for_exprs = ir_schedule_->GetLoops(Expr(sche_block_realize));
  std::vector<std::vector<Expr>> tiles(s_indices_.size() + r_indices_.size());

  VLOG(5) << "The number of loops to split in MultiLevelTiling is " << for_exprs.size();
  for (int i = for_exprs.size() - 1; i >= 0; --i) {
    ir::For* ir_for = for_exprs[i].As<ir::For>();
    VLOG(6) << "Applying Split for MultiLevelTiling on: " << Expr(ir_for);
    const std::vector<int>* idx = nullptr;
    if (sche_block->iter_vars[i]->is_reduce_axis) {
      idx = &r_indices_;
    } else {
      idx = &s_indices_;
    }  // TODO: support more iterator variable types

    int extent = ir_for->extent.as_int32();  // maybe int64?

    int num_split                      = idx->size();
    std::vector<int> tile_split_factor = SampleTileSplit<int>(extent, num_split);

    std::vector<Expr> splited = ir_schedule_->Split(Expr(ir_for), tile_split_factor);
    VLOG(6) << "Finish Split for MultiLevelTiling on above loop";
    for (int j = 0; j < num_split; ++j) {
      tiles[idx->at(j)].push_back(splited[j]);
    }
  }
  VLOG(5) << "Finish Split in MultiLevelTiling, before Reorder.";
  std::vector<Expr> splited_loops;
  for (auto& t : tiles) {
    std::reverse(t.begin(), t.end());
    splited_loops.insert(splited_loops.end(), t.begin(), t.end());
  }

  // Reorder has some bugs in this case. Enable it after we fix it.
  // ir_schedule_->Reorder(splited_loops);
  // VLOG(5) << "Finish Reorder in MultiLevelTiling";

  int num_binds = std::min(bind_axis_.size(), tiles.size());
  for (int i = 0; i < num_binds; ++i) {
    //   Enable fuse and then bind after fixing reorder bug.
    //   Expr fused = ir_schedule_->Fuse(tiles[i]);
    //   ir_schedule_->Bind(fused, bind_axis_[i]);
    ir_schedule_->Bind(tiles[i][0], bind_axis_[i]);
  }

  VLOG(4) << "Returning the result of MultiLevelTiling";
  return ir_schedule_->GetModule();
}

std::string MultiLevelTiling::GetRuleName() const { return "MultiLevelTiling"; }

AutoGenRule* MultiLevelTiling::NewPointer() const { return new MultiLevelTiling(*target_); }

}  // namespace auto_schedule
}  // namespace cinn

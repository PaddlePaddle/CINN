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
#include <unordered_map>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/analysis/analyze_ir.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
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

RuleApplyType MultiLevelTiling::Init(ir::IRSchedule* ir_schedule) {
  ir_schedule_        = ir_schedule;
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

void MultiLevelTiling::Apply(int index) {
  CHECK(ir_schedule_ != nullptr) << "Run MultiLevelTiling::Apply without Init";
  CHECK(num_applicable_ > 0 && applicable_indices_.size() == num_applicable_)
      << "MultiLevelTiling::Apply pre-condition doesn't meet";
  CHECK(index >= 0 && num_applicable_ > index)
      << "Invalid index for MultiLevelTiling::Apply, the index needs 0 <= index && index < NumberApplicable(), "
      << "Currently index = " << index << ",  NumberApplicable() = " << num_applicable_;

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

  // Have to GetLoops again because Split can change Block Expr(s)
  for_exprs = ir_schedule_->GetLoops(sche_block->name);
  std::unordered_map<std::string, int> loop_var_name_to_idx;
  for (int i = 0; i < for_exprs.size(); ++i) {
    loop_var_name_to_idx[for_exprs[i].As<ir::For>()->loop_var->name] = i;
  }
  CHECK(loop_var_name_to_idx.size() == for_exprs.size()) << "Loops contain duplicate loop var names after split";

  std::vector<Expr> splited_loops;
  for (auto& t : tiles) {
    std::reverse(t.begin(), t.end());
    for (auto& tile_loop_expr : t) {
      const ir::For* tile_loop = tile_loop_expr.As<ir::For>();
      CHECK(tile_loop) << "tiles store non For Expr";
      int idx = loop_var_name_to_idx[tile_loop->loop_var->name];
      splited_loops.push_back(for_exprs[idx]);
    }
  }

  ir_schedule_->Reorder(splited_loops);
  VLOG(5) << "Finish Reorder in MultiLevelTiling";

  int num_binds = std::min(bind_axis_.size(), tiles.size());
  for (int i = 0; i < num_binds; ++i) {
    loop_var_name_to_idx.clear();
    for_exprs = ir_schedule_->GetLoops(sche_block->name);
    for (int j = 0; j < for_exprs.size(); ++j) {
      loop_var_name_to_idx[for_exprs[j].As<ir::For>()->loop_var->name] = j;
    }
    CHECK(loop_var_name_to_idx.size() == for_exprs.size()) << "Loops contain duplicate loop var names before Fusion";
    for (int j = 0; j < tiles[i].size(); ++j) {
      const ir::For* tile_loop = tiles[i][j].As<ir::For>();
      CHECK(tile_loop) << "tiles store non For Expr";
      int idx     = loop_var_name_to_idx[tile_loop->loop_var->name];
      tiles[i][j] = for_exprs[idx];
    }

    Expr fused = ir_schedule_->Fuse(tiles[i]);
    ir_schedule_->Bind(fused, bind_axis_[i]);
    tiles[i] = {fused};
  }

  VLOG(4) << "Returning the result of MultiLevelTiling";
  return;
}

std::string MultiLevelTiling::GetRuleName() const { return "MultiLevelTiling"; }

}  // namespace auto_schedule
}  // namespace cinn

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
  return NeedsMultiLevelTiling(sche_block_realize);
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

  return num_applicable_ > 0 ? RuleApplyType::kApplyAndPruneOtherRules : RuleApplyType::kCannotApply;
}

void MultiLevelTiling::Apply(int index) {
  CHECK(ir_schedule_ != nullptr) << "Run MultiLevelTiling::Apply without Init";
  CHECK(num_applicable_ > 0 && applicable_indices_.size() == num_applicable_)
      << "MultiLevelTiling::Apply pre-condition doesn't meet";
  CHECK(index >= 0 && num_applicable_ > index)
      << "Invalid index for MultiLevelTiling::Apply, the index needs 0 <= index && index < NumberApplicable(), "
      << "Currently index = " << index << ",  NumberApplicable() = " << num_applicable_;

  int apply_index = applicable_indices_[index];
  Apply(ir_schedule_, all_block_realizes_[apply_index]);

  VLOG(4) << "Returning the result of MultiLevelTiling";
  return;
}

std::string MultiLevelTiling::GetRuleName() const { return "MultiLevelTiling"; }

RuleApplyType MultiLevelTiling::AnalyseApplyType(SearchState state, const std::string& block_name) const {
  Expr block_expr     = state->ir_schedule.GetBlock(block_name);
  auto* block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  CHECK(block_realize) << "stmt is not a ScheduleBlockRealize:" << block_expr;
  AnalyzeScheduleBlockReadWriteBuffer(block_realize->schedule_block.As<ir::ScheduleBlock>());

  return NeedsMultiLevelTiling(*block_realize) ? RuleApplyType::kApplyAndPruneOtherRules : RuleApplyType::kCannotApply;
}

std::vector<SearchState> MultiLevelTiling::ApplyOnBlock(SearchState state, const std::string& block_name) {
  SearchState new_state  = state.Copy();
  ir::IRSchedule* ir_sch = &new_state->ir_schedule;
  Expr block_expr        = ir_sch->GetBlock(block_name);
  Apply(ir_sch, block_expr);

  VLOG(4) << "Returning the result of MultiLevelTiling";
  return {new_state};
}

void MultiLevelTiling::Apply(ir::IRSchedule* ir_schedule, ir::Expr& block_expr) {
  ir::ScheduleBlockRealize* sche_block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sche_block                = sche_block_realize->schedule_block.As<ir::ScheduleBlock>();

  std::vector<Expr> for_exprs = ir_schedule->GetLoops(Expr(sche_block_realize));
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

    int num_split                       = idx->size();
    std::vector<Expr> tile_split_factor = ir_schedule->SamplePerfectTile(Expr(ir_for), num_split, 64);

    std::vector<Expr> splited = ir_schedule->Split(Expr(ir_for), tile_split_factor);
    VLOG(6) << "Finish Split for MultiLevelTiling on above loop";
    for (int j = 0; j < num_split; ++j) {
      tiles[idx->at(j)].push_back(splited[j]);
    }
  }
  VLOG(5) << "Finish Split in MultiLevelTiling, before Reorder.";

  // Have to GetLoops again because Split can change Block Expr(s)
  for_exprs = ir_schedule->GetLoops(sche_block->name);
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

  Expr reordered_expr = ir_schedule->Reorder(splited_loops);
  VLOG(5) << "Finish Reorder in MultiLevelTiling, now do Fuse and Binding on the main loop chain";

  int num_binds = std::min(bind_axis_.size(), tiles.size());
  for (int i = 0; i < num_binds; ++i) {
    loop_var_name_to_idx.clear();
    for_exprs = ir_schedule->GetLoops(sche_block->name);
    for (int j = 0; j < for_exprs.size(); ++j) {
      loop_var_name_to_idx[for_exprs[j].As<ir::For>()->loop_var->name] = j;
    }
    CHECK(loop_var_name_to_idx.size() == for_exprs.size()) << "Loops contain duplicate loop var names before Fusion";

    // Some loops extent may exceed the limited max factor (For example,
    // exceed the limit number of CUDA threads), here we check whether
    // the fused loop extent, which is the production of extends of loops
    // to be fused, is less or equal to the max factore.
    //
    // If yes, we fuse those loops and bind the fused loop
    // If no, we bind the first loop whose extent is less than the factor.
    int extent_prod                    = 1;
    int first_idx_less_than_max_factor = -1;
    for (int j = 0; j < tiles[i].size(); ++j) {
      const ir::For* tile_loop = tiles[i][j].As<ir::For>();
      CHECK(tile_loop) << "tiles store non For Expr";
      int idx     = loop_var_name_to_idx[tile_loop->loop_var->name];
      tiles[i][j] = for_exprs[idx];
      int extent  = tile_loop->extent.as_int32();  // maybe int64?
      extent_prod *= extent;
      if (first_idx_less_than_max_factor == -1 && extent <= max_factor_) {
        first_idx_less_than_max_factor = idx;
      }
    }

    if (extent_prod <= max_factor_) {
      Expr fused = ir_schedule->Fuse(tiles[i]);
      ir_schedule->Bind(fused, bind_axis_[i]);
    } else if (first_idx_less_than_max_factor != -1) {
      ir_schedule->Bind(for_exprs[first_idx_less_than_max_factor], bind_axis_[i]);
    }
  }

  VLOG(5) << "Do Fuse and Binding on the non-main loop chains";
  Expr sche_block_top_loop = ir_schedule->GetLoops(sche_block->name)[0];

  if (reordered_expr.As<ir::Block>()) {
    for (Expr& top_loop : reordered_expr.As<ir::Block>()->stmts) {
      if (top_loop != sche_block_top_loop) {
        std::vector<Expr> scan_loop_blocks = ir_schedule->GetAllBlocks();
        Expr other_loop_chain_schedule;
        for (Expr& block : scan_loop_blocks) {
          std::vector<Expr> loop_chain = ir_schedule->GetLoops(block);
          if (loop_chain[0] == top_loop) {
            other_loop_chain_schedule = block;
            break;
          }
        }
        if (!other_loop_chain_schedule.defined()) {
          LOG(WARNING) << "Has non-main loop chain, but not corresponding ScheduleBlock in MultiLevelTiling";
          continue;
        }

        std::string other_loop_schedule_name =
            other_loop_chain_schedule.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
        VLOG(6) << "Found other_loop_schedule_name = " << other_loop_schedule_name;
        int fuse_index = 0;
        for (int i = 0; i < num_binds; ++i) {
          for_exprs = ir_schedule->GetLoops(other_loop_schedule_name);

          // Some loops extent may exceed the limited max factor (For example,
          // exceed the limit number of CUDA threads), here we check whether
          // the fused loop extent, which is the production of extends of loops
          // to be fused, is less or equal to the max factore.
          //
          // If yes, we fuse those loops and bind the fused loop
          // If no, we bind the first loop whose extent is less than the factor.
          int extent_prod                    = 1;
          int first_idx_less_than_max_factor = -1;
          for (int j = 0; j < tiles[i].size(); ++j) {
            int extent = for_exprs[fuse_index + j].As<ir::For>()->extent.as_int32();
            extent_prod *= extent;
            if (first_idx_less_than_max_factor == -1 && extent <= max_factor_) {
              first_idx_less_than_max_factor = fuse_index + j;
            }
          }
          if (extent_prod <= max_factor_) {
            std::vector<Expr> loops_to_fuse(for_exprs.begin() + fuse_index,
                                            for_exprs.begin() + fuse_index + tiles[i].size());
            Expr fused = ir_schedule->Fuse(loops_to_fuse);
            ir_schedule->Bind(fused, bind_axis_[i]);
            fuse_index += 1;
          } else if (first_idx_less_than_max_factor != -1) {
            ir_schedule->Bind(for_exprs[first_idx_less_than_max_factor], bind_axis_[i]);
            fuse_index += tiles[i].size();
          }
        }
      }
    }
  }
}

}  // namespace auto_schedule
}  // namespace cinn

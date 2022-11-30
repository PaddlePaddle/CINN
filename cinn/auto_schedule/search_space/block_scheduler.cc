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

#include "cinn/auto_schedule/search_space/block_scheduler.h"

#include <algorithm>

#include "cinn/ir/ir.h"

namespace cinn {
namespace auto_schedule {

std::unique_ptr<BlockScheduler> BlockScheduler::Make(const std::vector<ir::Expr>& all_blocks,
                                                     const std::string& strategy,
                                                     const std::vector<int>& weights) {
  CHECK_GT(all_blocks.size(), 0) << "Empty block list";
  if (strategy == "traversal") {
    VLOG(6) << "Init TraversalBlockScheduler with block num = " << all_blocks.size();
    return std::make_unique<TraversalBlockScheduler>(all_blocks);
  } else if (strategy == "probabilistic") {
    VLOG(6) << "Init ProbabilisticBlockScheduler with block num = " << all_blocks.size();
    return std::make_unique<ProbabilisticBlockScheduler>(all_blocks, weights);
  }

  LOG(FATAL) << "Unimplementd strategy:" << strategy;
  return nullptr;
}

BlockScheduler::BlockScheduler(const std::vector<ir::Expr>& all_blocks) {
  std::transform(all_blocks.begin(), all_blocks.end(), std::back_inserter(all_blocks_), [](const ir::Expr& block_expr) {
    const ir::ScheduleBlockRealize* block_realize = block_expr.As<ir::ScheduleBlockRealize>();
    const ir::ScheduleBlock* block                = block_realize->schedule_block.As<ir::ScheduleBlock>();
    return block->name;
  });
}

std::string TraversalBlockScheduler::NextBlock() {
  if (cur_idx_ < all_blocks_.size()) {
    VLOG(6) << "[TraversalBlockScheduler] next block: " << all_blocks_.at(cur_idx_);
    return all_blocks_.at(cur_idx_++);
  }

  VLOG(6) << "[TraversalBlockScheduler] next block: empty";
  return "";
}

ProbabilisticBlockScheduler::ProbabilisticBlockScheduler(const std::vector<ir::Expr>& all_blocks,
                                                         const std::vector<int>& weights)
    : BlockScheduler(all_blocks) {
  if (weights.empty()) {
    for (int i = 0; i < all_blocks.size(); ++i) {
      cumulative_weight_.push_back(i + 1);
    }
  } else {
    CHECK_EQ(all_blocks.size(), weights.size());
    int cum = 0;
    for (int weight : weights) {
      cum += weight;
      cumulative_weight_.push_back(cum);
    }
  }
}

std::string ProbabilisticBlockScheduler::NextBlock() {
  int sample_index = rand() % cumulative_weight_.back();
  int block_idx =
      std::upper_bound(cumulative_weight_.begin(), cumulative_weight_.end(), sample_index) - cumulative_weight_.begin();

  VLOG(6) << "[ProbabilisticBlockScheduler] next block: " << all_blocks_.at(block_idx);
  return all_blocks_.at(block_idx);
}

}  // namespace auto_schedule
}  // namespace cinn
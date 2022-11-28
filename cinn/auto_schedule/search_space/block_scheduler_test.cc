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

#include <gtest/gtest.h>

#include "cinn/ir/ir.h"

namespace cinn {
namespace auto_schedule {

std::vector<ir::Expr> CreateTestBlocks() {
  std::vector<ir::Expr> blocks;
  for (int i = 0; i < 3; ++i) {
    ir::Expr block = ir::ScheduleBlock::Make({}, {}, {}, "block_" + std::to_string(i), ir::Expr());
    blocks.push_back(ir::ScheduleBlockRealize::Make({}, block));
  }
  return blocks;
}

TEST(BlockScheduler, Make) {
  std::vector<ir::Expr> mock_blocks(3);
  auto traversal_block_scheduler = BlockScheduler::Make(mock_blocks, "traversal");
  ASSERT_STREQ(traversal_block_scheduler->Name(), "traversal");
  auto probabilistic_block_scheduler = BlockScheduler::Make(mock_blocks, "probabilistic");
  ASSERT_STREQ(probabilistic_block_scheduler->Name(), "probabilistic");
}

TEST(TraversalBlockScheduler, NextBlock) {
  std::vector<ir::Expr> blocks   = CreateTestBlocks();
  auto traversal_block_scheduler = BlockScheduler::Make(blocks, "traversal");
  ASSERT_EQ("block_0", traversal_block_scheduler->NextBlock());
  ASSERT_EQ("block_1", traversal_block_scheduler->NextBlock());
  ASSERT_EQ("block_2", traversal_block_scheduler->NextBlock());
  ASSERT_EQ("", traversal_block_scheduler->NextBlock());
  traversal_block_scheduler->Reset();
  ASSERT_EQ("block_0", traversal_block_scheduler->NextBlock());
}

TEST(ProbabilisticBlockScheduler, NextBlock) {
  std::vector<ir::Expr> blocks       = CreateTestBlocks();
  auto probabilistic_block_scheduler = BlockScheduler::Make(blocks, "probabilistic", {4, 2, 1});
  std::string block_name;
  for (int i = 0; i < 20; ++i) {
    block_name = probabilistic_block_scheduler->NextBlock();
    VLOG(6) << "next block name: " << block_name;
  }
}

}  // namespace auto_schedule
}  // namespace cinn

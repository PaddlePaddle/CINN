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

#pragma once

#include <memory>
#include <vector>

#include "cinn/ir/ir_base.h"

namespace cinn {
namespace auto_schedule {

class SearchState;

class BlockScheduler {
 public:
  // Create a BlockScheduler with the specific strategy name
  // and necessary construct parameters.
  static std::unique_ptr<BlockScheduler> Make(const std::vector<ir::Expr>& all_blocks,
                                              const std::string& strategy     = "traversal",
                                              const std::vector<int>& weights = {});

  // Return the name of schedule strategy
  virtual const char* Name() const = 0;

  // Reset associated states to schedule at the beginning
  virtual void Reset() = 0;

  // Select a block to apply rule
  virtual std::string NextBlock() = 0;

 protected:
  // A BlockScheduler object should be created with the static function Make()
  BlockScheduler(const std::vector<ir::Expr>& all_blocks);

  // The names of all blocks
  // Because the Block Expr will be changed in the search process, the name is saved for indexing
  std::vector<std::string> all_blocks_;
};

// Schedule blocks with traversal strategy,
// witch means to select blocks one by one until all blocks are traversed.
class TraversalBlockScheduler : public BlockScheduler {
 public:
  TraversalBlockScheduler(const std::vector<ir::Expr>& all_blocks) : BlockScheduler(all_blocks), cur_idx_(0) {}

  const char* Name() const override { return "traversal"; }

  void Reset() override { cur_idx_ = 0; }

  std::string NextBlock() override;

 private:
  int cur_idx_;
};

// Schedule blocks with probabilistic strategy,
// witch means randomly picking blocks according to the given distribution.
class ProbabilisticBlockScheduler : public BlockScheduler {
 public:
  ProbabilisticBlockScheduler(const std::vector<ir::Expr>& all_blocks, const std::vector<int>& weights = {});

  const char* Name() const override { return "probabilistic"; }

  void Reset() override {}

  std::string NextBlock() override;

 private:
  std::vector<int> cumulative_weight_;
};

}  // namespace auto_schedule
}  // namespace cinn

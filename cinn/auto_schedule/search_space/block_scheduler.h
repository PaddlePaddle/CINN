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
  static std::unique_ptr<BlockScheduler> Make(const std::vector<ir::Expr>& all_blocks,
                                              const std::string& strategy     = "traversal",
                                              const std::vector<int>& weights = {});

  virtual const char* Name() const = 0;

  virtual void Reset() = 0;

  virtual std::string NextBlock() = 0;

 protected:
  BlockScheduler(const std::vector<ir::Expr>& all_blocks) : all_blocks_(&all_blocks) {}

  const std::vector<ir::Expr>* all_blocks_;
};

class TraversalBlockScheduler : public BlockScheduler {
 public:
  TraversalBlockScheduler(const std::vector<ir::Expr>& all_blocks) : BlockScheduler(all_blocks), cur_idx_(0) {}

  const char* Name() const override { return "traversal"; }

  void Reset() override { cur_idx_ = 0; }

  std::string NextBlock() override;

 private:
  int cur_idx_;
};

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

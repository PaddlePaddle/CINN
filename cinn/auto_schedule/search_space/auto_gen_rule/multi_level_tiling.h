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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/common/target.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class MultiLevelTiling : public AutoGenRule {
 public:
  MultiLevelTiling(const common::Target& target);
  ~MultiLevelTiling() = default;

  // initailize the AutoGenRule, it must be called before further actions.
  // Returns false if the rule cannot be applied on the mod_expr, true otherwise
  RuleApplyType Init(const ir::ModuleExpr& mod_expr) override;

  // Applies rule on the ir::ModuleExpr for a schedule block specified by index
  // between 0 (inclusive) and NumberApplicable() (exclusive)
  ir::ModuleExpr Apply(int index) override;

  // Returns the name of the rule, used for debug.
  std::string GetRuleName() const override;

  // Returns a pointer pointing to this rule. This class doesn't own the
  // pointer, caller should manage the life time of the pointer.
  AutoGenRule* NewPointer() const override;

  // Returns true if sche_block_realize is applicable by MultiLevelTiling
  bool MeetCondition(const ir::ScheduleBlockRealize& sche_block_realize) const;

  // Sample pair of integer type (a, b) such as a * b = extent
  template <typename T>
  std::vector<T> SampleSplitTwo(T extent) const {
    std::vector<std::vector<T>> candidates;
    for (T div = 1; div <= sqrt(extent); ++div) {
      if (extent % div == 0) {
        candidates.push_back({T(div), extent / div});
      }
    }
    if (candidates.size() == 0) {
      return {1, T(extent)};
    }
    int index           = rand() % candidates.size();
    std::vector<T> pick = candidates[index];
    if (rand() % 2 != 0) {
      T tmp   = pick[0];
      pick[0] = pick[1];
      pick[1] = tmp;
    }
    return pick;
  }

  // Sample num_split integers whose product equals extent
  template <typename T>
  std::vector<T> SampleTileSplit(T extent, int num_split) const {
    CHECK_GT(num_split, 0) << "num_split in SampleTileSplit must be greater than 0";
    if (num_split == 1) {
      return {extent};
    }
    std::vector<T> two_split = SampleSplitTwo<T>(extent);
    if (num_split == 2) {
      return two_split;
    }
    int half              = num_split >> 1;
    std::vector<T> result = SampleTileSplit<T>(two_split[0], half);
    std::vector<T> remind = SampleTileSplit<T>(two_split[1], num_split - half);
    result.insert(result.end(), remind.begin(), remind.end());
    return result;
  }

 private:
  std::unique_ptr<ir::IRSchedule> ir_schedule_;
  std::vector<ir::Expr> all_block_realizes_;
  std::vector<int> applicable_indices_;

  // Use char 'S' and 'R' to represent tile structure.
  // S means space tiling level and R means reduce tiling level
  //
  // For example, if tile_struct_ = "SSRSRS" and we are doing matrix
  // multiplication, i, j are the spatial indices and k is the reduce index,
  // the tiling result will be i_0, j0, i1, j1, k0, i2, j2, k1, i3, j3
  std::string tile_struct_;
  std::vector<int> s_indices_;
  std::vector<int> r_indices_;

  std::vector<std::string> bind_axis_;

  int max_factor = 64;
};

}  // namespace auto_schedule
}  // namespace cinn

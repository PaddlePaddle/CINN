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

#include "cinn/auto_schedule/search_strategy/evolutionary_search.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "cinn/auto_schedule/search_space/search_space.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/**
 * A mock search space is only used for test. It creates integer ir::Expr from
 * 0, -1, -2, ... and set the cost value same as the integer value.
 *
 * So evolutionary search should be able to find the minimal ModuleExpr with
 * smallest ir::Expr. This file tests it.
 */
class MockSearchSpace : public SearchSpace {
 public:
  MockSearchSpace(const TuneTask& tune_task) : SearchSpace(tune_task) {}

  int GetMinExprValue() const { return min_expr_value_; }

  int GetModuleExprSize() const { return module_expr_size_; }

  std::vector<ir::ModuleExpr> GetRandomInitialSketch(int num) override {
    std::vector<ir::ModuleExpr> ret;
    for (int i = 0; i < num; ++i) {
      std::vector<ir::Expr> exprs;
      for (int j = 0; j < module_expr_size_; ++j) {
        exprs.push_back(ir::Expr(-i));
      }
      min_expr_value_ = -i;
      ret.push_back(ir::ModuleExpr(exprs));
    }
    return ret;
  }

  std::pair<ir::ModuleExpr, float> GetScheduleMutate(const CostModel& cost_model,
                                                     const ir::ModuleExpr& mod_expr) override {
    float cost                  = 0.0f;
    std::vector<ir::Expr> exprs = mod_expr.GetExprs();
    for (const ir::Expr& expr : exprs) {
      cost += static_cast<float>((expr.as_int32()));
    }
    return std::make_pair<ir::ModuleExpr, float>(ir::ModuleExpr(exprs), float(cost));
  }

 private:
  int module_expr_size_ = 10;
  int min_expr_value_   = 0;
};

TEST(EvolutionarySearch, GetOneBest) {
  TuneTask mock_tune_task;
  EvolutionarySearch evolutionary_search;
  evolutionary_search.SetTuneTask(&mock_tune_task);

  MockSearchSpace* mock_search_space = new MockSearchSpace(mock_tune_task);
  // Ownership is transferred so don't delete mock_search_space
  evolutionary_search.SetSearchSpace(mock_search_space);

  ir::ModuleExpr best_mod_expr = evolutionary_search.SearchModuleExpr();

  std::vector<ir::Expr> exprs = best_mod_expr.GetExprs();
  EXPECT_GE(exprs.size(), 1UL);
  for (const ir::Expr& expr : exprs) {
    EXPECT_EQ(expr.as_int32(), mock_search_space->GetMinExprValue());
  }
}

TEST(EvolutionarySearch, GetEpsGreedy) {
  TuneTask mock_tune_task;
  EvolutionarySearch evolutionary_search(&mock_tune_task);

  MockSearchSpace* mock_search_space = new MockSearchSpace(mock_tune_task);
  // Ownership is transferred so don't delete mock_search_space
  evolutionary_search.SetSearchSpace(mock_search_space);
  std::vector<ir::ModuleExpr> mod_exprs = evolutionary_search.SearchModuleExprEpsGreedy();

  EXPECT_GE(mod_exprs.size(), 1UL);
  size_t expr_size = static_cast<size_t>(mock_search_space->GetModuleExprSize());
  for (const ir::ModuleExpr& mod_expr : mod_exprs) {
    EXPECT_EQ(mod_expr.GetExprs().size(), expr_size);
  }
}

}  // namespace auto_schedule
}  // namespace cinn

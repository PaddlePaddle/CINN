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

#include "cinn/ir/ir_compare.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/cinn.h"

namespace cinn {
namespace ir {

TEST(TestIrCompare, SingleFunction) {
  Target target = common::DefaultHostTarget();

  ir::Expr M(32);
  ir::Expr N(32);

  lang::Placeholder<float> A("A", {M, N});
  ir::Tensor B = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + ir::Expr(2.f); }, "B");
  ir::Tensor C = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + ir::Expr(2.f); }, "C");

  auto funcs_1 = lang::LowerVec("add_const", poly::CreateStages({A, B}), {A, B}, {}, {}, nullptr, target, true);
  auto funcs_2 = lang::LowerVec("add_const", poly::CreateStages({A, B}), {A, B}, {}, {}, nullptr, target, true);
  auto funcs_3 = lang::LowerVec("add_const", poly::CreateStages({A, C}), {A, C}, {}, {}, nullptr, target, true);

  ASSERT_EQ(funcs_1.size(), 1);
  ASSERT_EQ(funcs_2.size(), 1);
  ASSERT_EQ(funcs_3.size(), 1);

  IrEqualVistor compartor;
  // they are different at the name of root ScheduleBlock
  ASSERT_FALSE(compartor.Compare(funcs_1.front(), funcs_2.front()));
  // compare with itself
  ASSERT_TRUE(compartor.Compare(funcs_1.front(), funcs_1.front()));
  IrEqualVistor compartor_allow_suffix_diff(true);
  // they are euqal if allowing suffix of name different
  ASSERT_TRUE(compartor_allow_suffix_diff.Compare(funcs_1.front(), funcs_2.front()));

  ASSERT_FALSE(compartor.Compare(funcs_1.front(), funcs_3.front()));
  ASSERT_FALSE(compartor_allow_suffix_diff.Compare(funcs_1.front(), funcs_3.front()));
}

}  // namespace ir
}  // namespace cinn

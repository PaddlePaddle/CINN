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

#include <gtest/gtest.h>

#include "cinn/frontend/pass/test_helper.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"

namespace cinn::frontend {

TEST(DeadCodeEliminate, remove_single) {
  //              <x>
  //           /  | |   \
  //     identity | |  identity
  //             /   \
  //    reduce_sum  reduce_sum
  //          |         |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x            = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1   = builder.Identity(x);
  auto identity_2   = builder.Identity(x);
  auto reduce_sum_1 = builder.ReduceSum(x, {0, 1});
  auto reduce_sum_2 = builder.ReduceSum(x, {0, 1});

  PassTest tester;
  std::vector<std::string> input_names  = {x.id().data()};
  std::vector<std::string> output_names = {identity_1->id, reduce_sum_2->id};
  int num_removed_ops                   = tester.ApplyProgramPass(builder, {"DeadCodeEliminate"}, output_names);
  // identity_2, reduce_sum_1 and the corresponding instructions are removed.
  ASSERT_EQ(num_removed_ops, 2);
  tester.Execute(input_names, output_names);
}

TEST(DeadCodeEliminate, remove_multiple) {
  //              <x>
  //           /   |   \
  //     identity  |  reduce_sum
  //          \   /     |
  //           mul    <reduce_sum_1>
  //            |
  //         <mul_1>
  NetBuilder builder("net_builder");
  auto x            = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1   = builder.Identity(x);
  auto reduce_sum_1 = builder.ReduceSum(x, {0, 1});
  auto mul_1        = builder.Mul(x, identity_1);

  PassTest tester;
  std::vector<std::string> input_names  = {x.id().data()};
  std::vector<std::string> output_names = {reduce_sum_1->id};
  int num_removed_ops                   = tester.ApplyProgramPass(builder, {"DeadCodeEliminate"}, output_names);
  // Thus identity_1, mul_1 and the corresponding instructions are removed.
  ASSERT_EQ(num_removed_ops, 2);
  tester.Execute(input_names, output_names);
}

}  // namespace cinn::frontend

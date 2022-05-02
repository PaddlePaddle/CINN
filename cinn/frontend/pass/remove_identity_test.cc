// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include "cinn/frontend/program_pass.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn::frontend {

TEST(RemoveIdentity, can_remove) {
  //              <x>
  //           /       \
  //     identity   identity
  //          |         |
  //    reduce_sum  reduce_sum
  //          |         |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x            = builder.CreateInput(Float(32), {32, 16});
  auto identity_1   = builder.Identity(x);
  auto identity_2   = builder.Identity(x);
  auto reduce_sum_1 = builder.ReduceSum(x, {0, 1});
  auto reduce_sum_2 = builder.ReduceSum(x, {0, 1});

  PassTest tester;
  std::vector<std::string> input_names  = {x.id().data()};
  std::vector<std::string> output_names = {reduce_sum_1->id, reduce_sum_2->id};
  int num_moved_ops                     = tester.ApplyProgramPass(builder, {"RemoveIdentity"}, output_names);
  ASSERT_EQ(num_moved_ops, 2);
  tester.Execute(input_names, output_names);
}

TEST(RemoveIdentity, cant_remove_by_fetchids) {
  //              <x>
  //           /       \
  //     identity   identity
  //          |         |
  //    reduce_sum  reduce_sum
  //          |         |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x            = builder.CreateInput(Float(32), {32, 16});
  auto identity_1   = builder.Identity(x);
  auto identity_2   = builder.Identity(x);
  auto reduce_sum_1 = builder.ReduceSum(x, {0, 1});
  auto reduce_sum_2 = builder.ReduceSum(x, {0, 1});

  PassTest tester;
  std::vector<std::string> input_names  = {x.id().data()};
  std::vector<std::string> output_names = {identity_1->id, reduce_sum_1->id, reduce_sum_2->id};
  int num_moved_ops                     = tester.ApplyProgramPass(builder, {"RemoveIdentity"}, output_names);
  ASSERT_EQ(num_moved_ops, 1);
  tester.Execute(input_names, output_names);
}

TEST(RemoveIdentity, cant_remove_by_pattern) {
  //              <x>
  //           /   |   \
  //     identity  |  identity
  //          \   /     |
  //           mul    <identity_2>
  //            |
  //         <mul_1>
  NetBuilder builder("net_builder");
  auto x          = builder.CreateInput(Float(32), {32, 16});
  auto identity_1 = builder.Identity(x);
  auto identity_2 = builder.Identity(x);
  auto mul_1      = builder.Mul(x, identity_1);

  PassTest tester;
  std::vector<std::string> input_names  = {x.id().data()};
  std::vector<std::string> output_names = {mul_1->id};
  int num_moved_ops                     = tester.ApplyProgramPass(builder, {"RemoveIdentity"}, output_names);
  ASSERT_EQ(num_moved_ops, 1);
  tester.Execute(input_names, output_names);
}

}  // namespace cinn::frontend

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

#if 0
TEST(ReshapeRewriter, remove_single) {
  //              <x>
  //           /       \
  //     identity    reshape
  //          |         |
  //    reduce_sum  reduce_sum
  //          |         |
  // <reduce_sum_1> <reduce_sum_2>
  NetBuilder builder("net_builder");
  auto x            = builder.CreateInput(Float(32), {32, 16}, "x");
  auto identity_1   = builder.Identity(x);
  auto reshape_1    = builder.Reshape(x, {32, 16});
  auto reduce_sum_1 = builder.ReduceSum(identity_1, {0});
  auto reduce_sum_2 = builder.ReduceSum(reshape_1, {1});

  PassTest tester;
  std::vector<std::string> input_names  = {x.id().data()};
  std::vector<std::string> output_names = {reduce_sum_1->id, reduce_sum_2->id};
  int num_removed_ops = tester.ApplyProgramPass(builder, {"ReshapeRewriter", "RemoveIdentity"}, output_names);
  ASSERT_EQ(num_removed_ops, 2);
  tester.Execute(input_names, output_names);
}

TEST(ReshapeRewriter, remove_with_fill_constant) {
  //  fill_constant({16, 32})   <x>
  //          |                  |
  //     reshape({32, 16}     reshape
  //           \                /
  //             elementwise_add
  //                   |
  //                <add_1>
  NetBuilder builder("net_builder");
  auto x          = builder.CreateInput(Float(32), {32, 16}, "x");
  auto constant_1 = builder.FillConstant<float>({16, 32}, static_cast<float>(1.0), "constant_1");
  auto reshape_1  = builder.Reshape(constant_1, {32, 16});
  auto reshape_2  = builder.Reshape(x, {32, 16});
  auto add_1      = builder.ElementwiseAdd(reshape_1, reshape_2);

  PassTest tester;
  std::vector<std::string> input_names  = {x.id().data()};
  std::vector<std::string> output_names = {add_1->id};
  int num_removed_ops = tester.ApplyProgramPass(builder, {"ReshapeRewriter", "RemoveIdentity"}, output_names);
  ASSERT_EQ(num_removed_ops, 2);
  tester.Execute(input_names, output_names);
}
#endif

TEST(ReshapeRewriter, bugfix) {
  NetBuilder builder("net_builder");
  auto var_487 = builder.CreateInput(Float(32), {10201, 50}, "var_487");
  auto var_501 = builder.CreateInput(Float(32), {10201, 50}, "var_501");
  auto var_503 = builder.CreateInput(Float(32), {10201, 50}, "var_503");

  auto var_497 = builder.Tanh(var_487);  // fetch

  auto constant_1 = builder.FillConstant<float>({10201, 50}, static_cast<float>(1.0), "constant_1");
  auto var_513    = builder.ElementwiseMul(var_497, var_497);
  auto var_523    = builder.Sub(constant_1, var_513);  // fetch

  auto constant_2 = builder.FillConstant<float>({10201, 50}, static_cast<float>(1.0), "constant_2");
  auto var_509    = builder.ElementwiseMul(var_497, var_497);
  auto var_521    = builder.Sub(constant_2, var_509);  // fetch

  auto constant_3 = builder.FillConstant<float>({10201, 50}, static_cast<float>(1.0), "constant_3");
  auto var_517    = builder.ElementwiseMul(var_497, var_497);
  auto var_527    = builder.Sub(constant_3, var_517);  // fetch

  auto var_531 = builder.ElementwiseMul(var_501, var_523);  // fetch
  auto var_533 = builder.ElementwiseMul(var_503, var_523);  // fetch

  PassTest tester;
  std::vector<std::string> input_names  = {var_487.id().data(), var_501.id().data(), var_503.id().data()};
  std::vector<std::string> output_names = {
      var_497->id, var_523->id, var_521->id, var_527->id, var_531->id, var_533->id};
  int num_removed_ops = tester.ApplyProgramPass(builder, {"RemoveIdentity"}, output_names);
  tester.Execute(input_names, output_names);
}

}  // namespace cinn::frontend

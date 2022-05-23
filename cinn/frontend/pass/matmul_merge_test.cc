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

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/pass_test_helper.h"
#include "cinn/hlir/op/use_ops.h"

namespace cinn::frontend {

TEST(GemmRwriter, MergeTwoMatmul) {
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {2, 3}, "A");
  auto b       = builder.CreateInput(Float(32), {3, 4}, "B");
  auto x       = builder.Matmul(a, b);
  auto c       = builder.CreateInput(Float(32), {2, 3}, "C");
  auto d       = builder.CreateInput(Float(32), {3, 4}, "D");
  auto y       = builder.Matmul(c, d);
  auto out     = builder.Add(x, y);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), c.id(), d.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{{"Decomposer"}, {"MatmulMerge"}};
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123);
}

TEST(GemmRwriter, MergeTwoMatmulWithOneAncestor) {
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {2, 3}, "A");
  auto b       = builder.CreateInput(Float(32), {3, 3}, "B");
  auto x       = builder.Matmul(a, b);
  auto c       = builder.CreateInput(Float(32), {2, 3}, "C");
  auto d       = builder.CreateInput(Float(32), {3, 3}, "D");
  auto y       = builder.Matmul(c, d);
  auto e       = builder.CreateInput(Float(32), {3, 3}, "E");
  auto z       = builder.Matmul(y, e);
  auto out     = builder.Add(x, z);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), c.id(), d.id(), e.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{{"Decomposer"}, {"MatmulMerge"}};
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123);
}

TEST(GemmRwriter, MergeTwoGroupMatmul) {
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {2, 3}, "A");
  auto b = builder.CreateInput(Float(32), {3, 4}, "B");
  auto x = builder.Matmul(a, b);
  auto c = builder.CreateInput(Float(32), {4, 5}, "C");
  auto y = builder.Matmul(x, c);

  auto d = builder.CreateInput(Float(32), {2, 3}, "D");
  auto e = builder.CreateInput(Float(32), {3, 4}, "E");
  auto m = builder.Matmul(d, e);
  auto f = builder.CreateInput(Float(32), {4, 5}, "F");
  auto n = builder.Matmul(m, f);

  auto out     = builder.Add(y, n);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), c.id(), d.id(), e.id(), f.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{{"Decomposer"}, {"MatmulMerge"}};
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123);
}

}  // namespace cinn::frontend

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

#include "cinn/utils/functional.h"

#include <absl/algorithm/container.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ios>
#include <list>
#include <set>
#include <vector>

#include "cinn/utils/string.h"

namespace cinn {
namespace utils {

TEST(Functional, Flatten) {
  double s       = 3.14;
  auto flatten_s = Flatten(s);
  LOG(INFO) << utils::Join(flatten_s, ", ");
  ASSERT_EQ(flatten_s.size(), 1);
  ASSERT_TRUE(absl::c_equal(flatten_s, std::vector<double>{3.14}));

  std::vector<std::vector<int>> i{{3, 4, 5}, {7, 8, 9, 10}};
  auto flatten_i = Flatten(i);
  LOG(INFO) << utils::Join(flatten_i, ", ");
  ASSERT_EQ(flatten_i.size(), 7);
  ASSERT_TRUE(absl::c_equal(flatten_i, std::vector<int>{3, 4, 5, 7, 8, 9, 10}));

  std::vector<std::vector<std::vector<bool>>> v{{{true, false}, {true, false, true, false}},
                                                {{false}, {true, true, false}}};
  std::vector<bool> flatten_v = Flatten(v);
  LOG(INFO) << utils::Join(flatten_v, ", ");
  ASSERT_EQ(flatten_v.size(), 10);
  ASSERT_TRUE(
      absl::c_equal(flatten_v, std::vector<bool>{true, false, true, false, true, false, false, true, true, false}));

  std::list<std::set<std::vector<float>>> a{{{1, 2, 3}, {1, 2, 3, 4, 5, 6}}, {{1, 2.2f, 3}, {1, 2, 3.3f, 4.5f}}};
  auto flatten_a = Flatten(a);
  LOG(INFO) << utils::Join(flatten_a, ", ");
  ASSERT_EQ(flatten_a.size(), 16);
  ASSERT_TRUE(absl::c_equal(flatten_a, std::vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3.3, 4.5, 1, 2.2, 3}));

  std::list<std::list<std::vector<int>>> b;
  auto flatten_b = Flatten(b);
  LOG(INFO) << utils::Join(flatten_b, ", ");
  ASSERT_EQ(flatten_b.size(), 0);
  ASSERT_TRUE(absl::c_equal(flatten_b, std::vector<int>{}));
}

}  // namespace utils
}  // namespace cinn

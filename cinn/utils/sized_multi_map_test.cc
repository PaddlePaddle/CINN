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

#include "cinn/utils/sized_multi_map.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace cinn {
namespace utils {

TEST(SizedMultiMap, PopMax) {
  SizedMultiMap<int, std::string> sized_multi_map(5);

  for (int i = 0; i < 10; ++i) {
    sized_multi_map.Push(i, std::to_string(i));
    if (i < 5) {
      EXPECT_EQ(sized_multi_map.Size(), static_cast<size_t>(i + 1));
      auto max_key_value = std::pair<int, std::string>(i, std::to_string(i));
      EXPECT_EQ(sized_multi_map.MaxKeyValuePair(), max_key_value);
      auto min_key_value = std::pair<int, std::string>(0, "0");
      EXPECT_EQ(sized_multi_map.MinKeyValuePair(), min_key_value);
    } else {
      EXPECT_EQ(sized_multi_map.Size(), 5UL);
      auto max_key_value = std::pair<int, std::string>(4, "4");
      EXPECT_EQ(sized_multi_map.MaxKeyValuePair(), max_key_value);
      auto min_key_value = std::pair<int, std::string>(0, "0");
      EXPECT_EQ(sized_multi_map.MinKeyValuePair(), min_key_value);
    }
  }

  std::vector<std::pair<int, std::string>> vec =
      sized_multi_map.ReturnAsContainer<std::vector<std::pair<int, std::string>>>();
  for (int i = 0; i < 5; ++i) {
    auto key_value = std::pair<int, std::string>(i, std::to_string(i));
    EXPECT_EQ(vec[i], key_value);
  }

  for (int i = 0; i < 4; ++i) {
    sized_multi_map.Pop();
    EXPECT_EQ(sized_multi_map.Size(), static_cast<size_t>(4 - i));
    auto max_key_value = std::pair<int, std::string>(3 - i, std::to_string(3 - i));
    EXPECT_EQ(sized_multi_map.MaxKeyValuePair(), max_key_value);
    auto min_key_value = std::pair<int, std::string>(0, "0");
    EXPECT_EQ(sized_multi_map.MinKeyValuePair(), min_key_value);
  }
}

TEST(SizedMultiMap, PopMin) {
  SizedMultiMap<int, std::string> sized_multi_map(5, /* pop_max_when_full = */ false);
  for (int i = 0; i < 10; ++i) {
    sized_multi_map.Push(i, std::to_string(i));
    if (i < 5) {
      EXPECT_EQ(sized_multi_map.Size(), static_cast<size_t>(i + 1));
      auto max_key_value = std::pair<int, std::string>(i, std::to_string(i));
      EXPECT_EQ(sized_multi_map.MaxKeyValuePair(), max_key_value);
      auto min_key_value = std::pair<int, std::string>(0, "0");
      EXPECT_EQ(sized_multi_map.MinKeyValuePair(), min_key_value);
    } else {
      EXPECT_EQ(sized_multi_map.Size(), 5UL);
      auto max_key_value = std::pair<int, std::string>(i, std::to_string(i));
      EXPECT_EQ(sized_multi_map.MaxKeyValuePair(), max_key_value);
      auto min_key_value = std::pair<int, std::string>(i - 4, std::to_string(i - 4));
      EXPECT_EQ(sized_multi_map.MinKeyValuePair(), min_key_value);
    }
  }

  std::vector<std::pair<int, std::string>> vec =
      sized_multi_map.ReturnAsContainer<std::vector<std::pair<int, std::string>>>();
  for (int i = 0; i < 5; ++i) {
    auto key_value = std::pair<int, std::string>(i + 5, std::to_string(i + 5));
    EXPECT_EQ(vec[i], key_value);
  }

  for (int i = 0; i < 4; ++i) {
    sized_multi_map.Pop();
    EXPECT_EQ(sized_multi_map.Size(), static_cast<size_t>(4 - i));
    auto max_key_value = std::pair<int, std::string>(9, "9");
    EXPECT_EQ(sized_multi_map.MaxKeyValuePair(), max_key_value);
    auto min_key_value = std::pair<int, std::string>(6 + i, std::to_string(6 + i));
    EXPECT_EQ(sized_multi_map.MinKeyValuePair(), min_key_value);
  }
}

}  // namespace utils
}  // namespace cinn

// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "cinn/common/axis.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>

#include "cinn/utils/string.h"

namespace cinn {
namespace common {

TEST(AXISNAME, BASE) {
  ASSERT_EQ(axis_name(0), std::string("i"));
  ASSERT_EQ(axis_name(1), std::string("j"));
  ASSERT_EQ(axis_name(22), std::string("ii"));
  ASSERT_EQ(axis_name(44), std::string("iii"));
}

TEST(AXISNAME, CHECK_RESERVED) {
  ASSERT_TRUE(IsAxisNameReserved("i"));
  ASSERT_TRUE(IsAxisNameReserved("j"));
  ASSERT_TRUE(IsAxisNameReserved("ii"));
  ASSERT_TRUE(IsAxisNameReserved("iiiiiiiiii"));
  ASSERT_FALSE(IsAxisNameReserved("ijk"));
  ASSERT_FALSE(IsAxisNameReserved("iiiiiiiiiij"));
  ASSERT_FALSE(IsAxisNameReserved("x"));
}

}  // namespace common
}  // namespace cinn

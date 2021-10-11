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

#include "cinn/poly/ast_gen.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace poly {

TEST(TransIdentityExtentToContextId, basic) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl::set set(ctx, "{ s[i,j=0,k] : 0<=i<12 and 12<k<32 }");
  auto new_set = TransIdentityExtentToContextId(set);
  LOG(INFO) << new_set;

  ASSERT_EQ(utils::GetStreamCnt(new_set),
            "[_const_0] -> { s[i, j, k] : _const_0 <= 1 and 0 <= i <= 11 and 0 <= j <= _const_0 and 13 <= k <= 31 }");
}

TEST(TransIdentityExtentToContextId, basic1) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl::set set(ctx, "[n] -> { s[i,j=0,k] : 0<=i<n and 12<k<32 }");
  LOG(INFO) << "set: " << set;
  auto new_set = TransIdentityExtentToContextId(set);
  LOG(INFO) << new_set;
}

}  // namespace poly
}  // namespace cinn

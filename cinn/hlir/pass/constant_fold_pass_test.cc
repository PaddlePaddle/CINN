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

#include <gtest/gtest.h>

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace hlir {
namespace pass {

TEST(Constant_Folding, fold_broadcast_to_const_scalar) {}

TEST(Constant_Folding, fold_broadcast_to_fill_constant) {}

TEST(Constant_Folding, fold_reshape_fill_constant) {}

TEST(Constant_Folding, fold_squeeze_fill_constant) {}

TEST(Constant_Folding, fold_expand_dims_to_fill_constant) {}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

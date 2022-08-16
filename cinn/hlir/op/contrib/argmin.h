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

#pragma once
#include <string>
#include <vector>

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace op {
/**
 * @brief find the argmin of array elements over a given axis
 *
 * @param A The input Tensor
 * @param axis Axis or axes to find the argmin over. If axis is empty, the operation will product over all elements of
 * the input array. If axis is negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in the result as dimensions with size one.
 * With this option, the result will broadcast correctly against the input array.
 * @param stages The stage map
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor Argmin(const ir::Tensor& A,
                  const int& axis,
                  const bool keep_dims,
                  poly::StageMap stages,
                  const std::string& output_name = "T_Argmin_out");
}  // namespace op
}  // namespace hlir
}  // namespace cinn

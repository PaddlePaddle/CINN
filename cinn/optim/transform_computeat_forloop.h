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

#pragma once
#include <algorithm>
#include <unordered_set>
#include <utility>

#include "cinn/ir/ir.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace optim {

using forloop_infos_t = std::map<std::string, std::map<std::string, poly::StageForloopInfo>>;

/**
 * In X86 ends, remove the loop iterators within the compute_at level in temp tensor's indices and resize its shape
 * correspondingly.
 *
 * For example, A_write_cache->ComputeAt(A, 1)
 * \code
 * for (i, 0, 10)
 *   for (j, 0, 10)
 *      for (a, 0, 10)
 *          A_write_cache(i,j,a) = i * j * a
 *      for (k, 0, 10)
 *          A(i,j,k) = A_write_cache(i,j,k)
 * \endcode
 *
 * will be replaced to
 *
 * \code
 * for (i, 0, 10)
 *   for (j, 0, 10)
 *      for (a, 0, 10)
 *          A_write_cache(a) = i * j * a
 *      for (k, 0, 10)
 *          A(i,j,k) = A_write_cache(k)
 * \endcode
 *
 * And the temp tensor A_write_cache's shape will be resized from (10*10*10) to 10.
 *
 * @param expr The expression to modify.
 * @param global_tensor_map The map mapping a tensor's name to itself.
 * @param forloop_infos A map of forloop to their infomation.
 */
void TransformComputeatForloops(const forloop_infos_t& forloop_infos,
                                const std::vector<std::string>& traverse_order,
                                std::map<std::string, ir::Tensor>* global_tensor_map,
                                std::unordered_set<std::string>& resized_buffer,
                                Expr* expr);

}  // namespace optim
}  // namespace cinn

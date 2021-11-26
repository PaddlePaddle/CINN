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
#include <string>
#include <vector>

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace pe {
/**
 * @brief sums array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes along which a sum is performed. If axis is empty, the operation will sum over all elements
 * of the input array. If axis is negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in the result as dimensions with size one.
 * With this option, the result will broadcast correctly against the input array.
 * @param initial Starting value for the sum.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensors.
 */
ir::Tensor ReduceSum(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     bool keep_dims                 = false,
                     Expr initial                   = Expr(0.f),
                     const std::string& output_name = "T_Reduce_Sum_out");

/**
 * @brief product array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes along which a production is performed. If axis is empty, the operation will product over all
 * elements of the input array. If axis is negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in the result as dimensions with size one.
 * With this option, the result will broadcast correctly against the input array.
 * @param initial Starting value for the production.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensors.
 */
ir::Tensor ReduceProd(const ir::Tensor& A,
                      const std::vector<int>& axis,
                      bool keep_dims                 = false,
                      Expr initial                   = Expr(1.f),
                      const std::string& output_name = "T_Reduce_Prod_out");

/**
 * @brief find the maxium of array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes to find the maximum over. If axis is empty, the operation will product over all elements of
 * the input array. If axis is negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in the result as dimensions with size one.
 * With this option, the result will broadcast correctly against the input array.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor ReduceMax(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     bool keep_dims                 = false,
                     Expr initial                   = Expr(),
                     const std::string& output_name = "T_Reduce_Max_out");

/**
 * @brief find the minimum of array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes to find the minimum over. If axis is empty, the operation will product over all elements of
 * the input array. If axis is negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in the result as dimensions with size one.
 * With this option, the result will broadcast correctly against the input array.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor ReduceMin(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     bool keep_dims                 = false,
                     Expr initial                   = Expr(),
                     const std::string& output_name = "T_Reduce_Min_out");

/**
 * @brief find the max of array elements over the last dimension
 *
 * @param A The input Tensor
 * @param output_name The name of the output Tensor
 */
std::vector<ir::Tensor> WarpReduceMax(const ir::Tensor& A,
                                      int last_reduce_dim,
                                      const std::string& output_name = "T_Warp_Reduce_Max_out");

/**
 * @brief compute the sum of array elements over the last dimension
 *
 * @param A The input Tensor
 * @param output_name The name of the output Tensor
 */
std::vector<ir::Tensor> WarpReduceSum(const ir::Tensor& A,
                                      int last_reduce_dim,
                                      const std::string& output_name = "T_Warp_Reduce_Sum_out");

/**
 * @brief compute the average of array elements over the last dimension
 *
 * @param A The input Tensor
 * @param output_name The name of the output Tensor
 */
std::vector<ir::Tensor> WarpReduceAvg(const ir::Tensor& A,
                                      int last_reduce_dim,
                                      const std::string& output_name = "T_Warp_Reduce_Avg_out");

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

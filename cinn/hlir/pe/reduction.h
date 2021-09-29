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

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

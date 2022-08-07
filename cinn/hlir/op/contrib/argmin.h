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
 * @param stages The stage map
 * @param axis Axis or axes to find the argmin over. If axis is empty, the operation will product over all elements of
 * the input array. If axis is negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in the result as dimensions with size one.
 * With this option, the result will broadcast correctly against the input array.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor Argmin(const ir::Tensor& A,
                  const int& axis,
                  const bool keep_dims           = false,
                  const std::string& output_name = "T_Argmin_out");
}  // namespace pe
}  // namespace hlir
}  // namespace cinn

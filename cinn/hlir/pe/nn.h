#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace pe {

/**
 * @brief Rectified Linear Unit.
 *
 * @param A The first Tensor
 * @param threshold The relu threshold (default: 0)
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
template <typename T>
ir::Tensor Relu(const ir::Tensor& A, T threshold = static_cast<T>(0), const std::string& output_name = "T_Relu_out");
/**
 * @brief Leaky Rectified Linear Unit.
 *
 * @param A The first Tensor
 * @param alpha The slope for the small gradient when t < 0
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor LeakyRelu(const ir::Tensor& A, double alpha = 0.1, const std::string& output_name = "T_LeakyRelu_out");
/**
 * @brief Leaky Rectified Linear Unit.
 *
 * @param A The first Tensor
 * @param slope The channel-wise slope tensor
 * @param axis The axis where the channel data needs to be applied
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor PRelu(const ir::Tensor& A,
                 const ir::Tensor& slope,
                 const int axis                 = 1,
                 const std::string& output_name = "T_PRelu_out");

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

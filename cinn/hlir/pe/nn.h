#pragma once

#include <string>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/lang/compute.h"

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
ir::Tensor Relu(const ir::Tensor& A, T threshold = static_cast<T>(0), const std::string& output_name = "T_Relu_out") {
  return lang::Compute(
      A->shape, [&](const std::vector<Expr>& indice) { return ir::Relu(A(indice), threshold); }, output_name);
}

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

std::vector<ir::Tensor> Conv2d_NCHW(const ir::Tensor& input,
                                    const ir::Tensor& weights,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation,
                                    int groups,
                                    const std::string& output_name);

ir::Tensor BatchNorm_NCHW(const ir::Tensor& input,
                          const ir::Tensor& weights,
                          float epsilon,
                          const std::string& output_name);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

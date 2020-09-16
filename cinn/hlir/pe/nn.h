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
 * @brief Rectified Linear Unit bounded by six.
 *
 * @param A The first Tensor
 * @param threshold The relu threshold (default: 0)
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
template <typename T>
ir::Tensor Relu6(const ir::Tensor& A, T threshold = static_cast<T>(0), const std::string& output_name = "T_Relu6_out") {
  return lang::Compute(
      A->shape, [&](const std::vector<Expr>& indice) { return ir::Relu6(A(indice), threshold); }, output_name);
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

std::vector<ir::Tensor> Softmax(const ir::Tensor& A, int axis, const std::string& output_name);
/**
 * @brief Perform pooling on the width dimension of the tensor.
 *        Width axis is determined by the data_format string in which 'W' means width. Only support NCW and NWC
 * data_format.
 * @param tensor The input tensor with shape of {N, C, W} or {N, W, C}
 * @param kernel_size Vector of ints: {pool_kernel_width}
 * @param stride_size Vector of ints: {pool_stride_width}
 * @param padding_size Vector of ints: {head_pad_width, tail_pad_width}
 * @param pool_type The type of pooling operator, currently support "max" and "avg". Default is "max".
 * @param ceil_mode Whether to use ceil when calculating the output size. Default is false.
 * @param exclusive Whether include padding in the calculation. Default is True.
 * @param data_format The input data format. Only support NCW and NWC data_format.
 * @param output_name the name of the output tensor after padding and pooling.
 *
 * @return the vector of padding tensor and pooling tensor.
 */
std::vector<ir::Tensor> Pool1d(const ir::Tensor& tensor,
                               const std::vector<int>& kernel_size,
                               const std::vector<int>& stride_size,
                               const std::vector<int>& padding_size,
                               const std::string& pool_type   = "max",
                               bool ceil_mode                 = false,
                               bool exclusive                 = true,
                               const std::string& data_format = "NCW",
                               const std::string& output_name = "T_Pool1d_out");

/**
 * @brief Perform pooling on the height and width dimension of the tensor.
 *        Height and width axes are determined by the data_format string in which 'H' means height and 'W' means width.
 *        Only support NCHW and NHWC data_format.
 * @param tensor The input tensor with shape of {N, C, H, W} or {N, H, W, C}
 * @param kernel_size Vector of ints: {pool_kernel_height, pool_kernel_width}
 * @param stride_size Vector of ints: {pool_stride_height, pool_stride_width}
 * @param padding_size Vector of ints: {head_pad_height, head_pad_width, tail_pad_height, tail_pad_width}
 * @param pool_type The type of pooling operator, currently support "max" and "avg". Default is "max".
 * @param ceil_mode Whether to use ceil when calculating the output size. Default is false.
 * @param exclusive Whether include padding in the calculation. Default is True.
 * @param data_format The input data format. Only support NCHW and NHWC data_format.
 * @param output_name the name of the output tensor after padding and pooling.
 *
 * @return the vector of padding tensor and pooling tensor.
 */
std::vector<ir::Tensor> Pool2d(const ir::Tensor& tensor,
                               const std::vector<int>& kernel_size,
                               const std::vector<int>& stride_size,
                               const std::vector<int>& padding_size,
                               const std::string& pool_type   = "max",
                               bool ceil_mode                 = false,
                               bool exclusive                 = true,
                               const std::string& data_format = "NCHW",
                               const std::string& output_name = "T_Pool2d_out");

/**
 * @brief Perform pooling on the depth, height and width dimension of the tensor.
 *        Depth, height and width axis is determined by the data_format string in which 'D' means depth, 'H' means
 * height and 'W' means width. Only support NCDHW and NDHWC data_format.
 * @param tensor The input tensor with shape of {N, C, D, H, W} or {N, D, H, W, C}
 * @param kernel_size Vector of ints: {pool_kernel_depth, pool_kernel_height, pool_kernel_width}
 * @param stride_size Vector of ints: {pool_stride_depth, pool_stride_height, pool_stride_width}
 * @param padding_size Vector of ints: {head_pad_depth, head_pad_height, head_pad_width, tail_pad_depth,
 * tail_pad_height, tail_pad_width}
 * @param pool_type The type of pooling operator, currently support "max" and "avg". Default is "max".
 * @param ceil_mode Whether to use ceil when calculating the output size. Default is false.
 * @param exclusive Whether include padding in the calculation. Default is True.
 * @param data_format The input data format. Only support NCDHW and NDHWC data_format.
 * @param output_name the name of the output tensor after padding and pooling.
 */
std::vector<ir::Tensor> Pool3d(const ir::Tensor& x,
                               const std::vector<int>& kernel_size,
                               const std::vector<int>& stride_size,
                               const std::vector<int>& padding_size,
                               const std::string& pool_type   = "max",
                               bool ceil_mode                 = false,
                               bool exclusive                 = true,
                               const std::string& data_format = "NCDHW",
                               const std::string& output_name = "T_Pool3d_out");

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

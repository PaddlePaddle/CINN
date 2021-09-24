#pragma once

#include <string>
#include <unordered_map>

#include "cinn/frontend/symbolization/base_builder.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace symbolization {
class CoarseBuilder : public BaseBuilder {
 public:
  using BaseBuilder::BaseBuilder;

  /**
   * Add two variables.
   *
   * @param a The first variable.
   * @param b The second variable.
   * @return The result.
   */
  Variable add(const Variable& a, const Variable& b);

  /**
   * Multiply two matrix.
   */
  Variable mul(const Variable& a, const Variable& b, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * Multiply two matrix and add a bias.
   */
  Variable mulbias(
      const Variable& a, const Variable& b, const Variable& c, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * Add two tensors element-wise.
   */
  Variable elementwise_add(const Variable& a, const Variable& b, int axis = -1);

  /**
   * Multiply two tensors element-wise.
   */
  Variable elementwise_mul(const Variable& a, const Variable& b, int axis = -1);

  /**
   * Apply Rectified Linear Unit on input Variable.
   * Actually apply: outupt = max(input,0)
   *
   * @param a The first variable.
   * @return The result.
   */
  Variable relu(const Variable& a);

  Variable relu6(const Variable& a);

  /**
   * The convolution2D layer calculates the output based on the input, filter
   * and strides, paddings, dilations, groups parameters.
   *
   * @param a The first variable input.
   * @param b The second variable filter(weights).
   * @param attr_store The params like padding, stride, dilation, etc.
   * @return The result.
   */
  Variable conv2d(const Variable& a, const Variable& b, const std::unordered_map<std::string, AttrT>& attr_store);

  Variable layout_transform(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store);

  Variable conv2d_NCHWc(const Variable& a, const Variable& b, const std::unordered_map<std::string, AttrT>& attr_store);

  Variable depthwise_conv2d(const Variable& a,
                            const Variable& b,
                            const std::unordered_map<std::string, AttrT>& attr_store);

  Variable pool2d(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store);

  /**
   * The batchnorm layer can be used as a normalizer function
   * for convolution or fully_connected operations.
   *
   * @param a The first variable input.
   * @param b The second variable filter(weights).
   * @param attr_store The params like eplison.
   * @return The result.
   */
  Variable batchnorm(const Variable& a,
                     const Variable& scale,
                     const Variable& bias,
                     const Variable& mean,
                     const Variable& variance,
                     const std::unordered_map<std::string, AttrT>& attr_store);

  Variable scale(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store);

  Variable softmax(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store);

  Variable sigmoid(const Variable& a);

  Variable slice(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store);

  Variable dropout_infer(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store);
};
}  // namespace symbolization
}  // namespace frontend
}  // namespace cinn

#pragma once

#include <string>
#include <vector>

#include "cinn/frontend/base_builder.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

class NetBuilder : public BaseBuilder {
 public:
  using BaseBuilder::BaseBuilder;

  /**
   * Add two variables.
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
   */
  Variable relu(const Variable& a);

  Variable relu6(const Variable& a, float threshold = 6.0f);

  /**
   * The convolution2D layer calculates the output based on the input, filter
   * and strides, paddings, dilations, groups parameters.
   */
  Variable conv2d(const Variable& a,
                  const Variable& b,
                  const std::vector<int>& strides      = {1, 1},
                  const std::vector<int>& paddings     = {0, 0},
                  const std::vector<int>& dilations    = {1, 1},
                  int groups                           = 1,
                  const std::string& data_format       = "NCHW",
                  const std::string& padding_algorithm = "EXPLICIT");

  Variable depthwise_conv2d(const Variable& a,
                            const Variable& b,
                            const std::vector<int>& strides      = {1, 1},
                            const std::vector<int>& paddings     = {0, 0},
                            const std::vector<int>& dilations    = {1, 1},
                            int groups                           = 1,
                            const std::string& data_format       = "NCHW",
                            const std::string& padding_algorithm = "EXPLICIT");

  Variable pool2d(const Variable& a,
                  const std::string& pooling_type,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides      = {1, 1},
                  const std::vector<int>& paddings     = {0, 0},
                  bool ceil_mode                       = false,
                  bool exclusive                       = true,
                  bool global_pooling                  = false,
                  const std::string& data_format       = "NCHW",
                  bool adaptive                        = false,
                  const std::string& padding_algorithm = "EXPLICIT");

  /**
   * The batchnorm layer can be used as a normalizer function
   * for convolution or fully_connected operations.
   */
  Variable batchnorm(const Variable& a,
                     const Variable& scale,
                     const Variable& bias,
                     const Variable& mean,
                     const Variable& variance,
                     float epsilon                  = 1e-5f,
                     float momentum                 = 0.9f,
                     const std::string& data_layout = "NCHW");

  Variable scale(const Variable& a, float scale = 1.0f, float bias = 0.0f, bool bias_after_scale = true);

  Variable softmax(const Variable& a, int axis = -1, const std::string& data_format = "AnyLayout");

  Variable sigmoid(const Variable& a);

  Variable slice(const Variable& a,
                 const std::vector<int>& axes,
                 const std::vector<int>& starts        = {},
                 const std::vector<int>& ends          = {},
                 const std::vector<int>& infer_flags   = {},
                 const std::vector<int>& decrease_axis = {});

  Variable dropout_infer(const Variable& a,
                         float dropout_prob                        = 0.5f,
                         const std::string& dropout_implementation = "downgrade_in_infer");
};

}  // namespace frontend
}  // namespace cinn

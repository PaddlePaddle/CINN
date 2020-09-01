#include "cinn/hlir/pe/nn.h"

#include <string>
#include <vector>

#include "cinn/common/context.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/optim/ir_simplify.h"

namespace cinn {
namespace hlir {
namespace pe {

using cinn::lang::Compute;
using namespace ir;

enum PoolType {
  kAvgPool,
  kMaxPool,
};

Tensor LeakyRelu(const Tensor &A, double alpha, const std::string &output_name) {
  return Compute(
      A->shape, [&](const std::vector<Expr> &indice) { return LeakyRelu(A(indice), alpha); }, output_name);
}

Tensor PRelu(const Tensor &A, const Tensor &slope, const int axis, const std::string &output_name) {
  CHECK_LT(axis, A->shape.size()) << "Wrong axis value: " << axis << std::endl;
  CHECK(A->shape[axis] == slope->shape[0]) << "Wrong slope shape: " << slope->shape[0] << std::endl;
  return Compute(
      A->shape,
      [&](const std::vector<Expr> &indice) { return LeakyRelu(A(indice), slope(indice[axis])); },
      output_name);
}

std::vector<ir::Tensor> Conv2d_NCHW(const ir::Tensor &input,
                                    const ir::Tensor &weights,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation,
                                    int groups,
                                    const std::string &output_name) {
  CHECK_EQ(4, input->shape.size()) << "Input's dimension of Conv2d op is not 4! Please check.";
  CHECK_EQ(4, weights->shape.size()) << "Weight's dimension of Conv2d op is not 4! Please check.";
  std::vector<Expr> output_shape{
      input->shape[0],                                                                                // B
      weights->shape[0],                                                                              // O
      Expr((input->shape[2] - ((weights->shape[2] - 1) * dilation + 1) + 2 * pad_h) / stride_h + 1),  // H
      Expr((input->shape[3] - ((weights->shape[3] - 1) * dilation + 1) + 2 * pad_w) / stride_w + 1)   // W
  };
  auto input_pad = Compute(
      {input->shape[0], input->shape[1], input->shape[2] + 2 * pad_h, input->shape[3] + 2 * pad_w},
      [=](Expr nn, Expr cc, Expr yy, Expr xx) {
        auto cond =
            ir::logic_and({yy >= pad_h, yy - pad_h < input->shape[2], xx >= pad_w, xx - pad_w < input->shape[3]});
        return ir::Select::Make(cond, input(nn, cc, yy - pad_h, xx - pad_w), Expr(0.f));
      },
      UniqName("input_pad"));
  std::vector<Expr> new_weights_shape{weights->shape[0],
                                      weights->shape[1],
                                      dilation * (weights->shape[2] - 1) + 1,
                                      dilation * (weights->shape[3] - 1) + 1};
  auto weights_dilation = Compute(
      new_weights_shape,
      [=](Expr nn, Expr cc, Expr yy, Expr xx) {
        auto cond = ir::logic_and({(xx) % dilation == 0, yy % dilation == 0});
        return ir::Select::Make(cond, weights(nn, cc, yy / dilation, xx / dilation), Expr(0.f));
      },
      UniqName("weights_dilation"));

  Var rc(input_pad->shape[1], UniqName("rc"));
  Var ry(weights_dilation->shape[2], UniqName("ry"));
  Var rx(weights_dilation->shape[3], UniqName("rx"));

  auto res = Compute(output_shape,
                     [=](Expr nn, Expr ff, Expr yy, Expr xx) {
                       return ir::ReduceSum(
                           input_pad(nn, rc, yy * stride_h + ry, xx * stride_w + rx) * weights_dilation(ff, rc, ry, rx),
                           Expr(0.f));
                     },
                     output_name,
                     {ry, rx, rc});
  return {input_pad, weights_dilation, res};
}

/**
 * Can be used as a normalizer function for convolution or fully_connected operations.
 * Specified for NCHW layout.
 * Math: Y = (X - mean) / sqrt(variance + epsilon) * scale + bias
 * @param input The input variable.
 * @param weights The weights containing mean, variance, scale and bias.
 * @param epsilon The param epsilon is added to avoid divide zero.
 * @param output_name The name of output tensor.
 * @return The calculated output tensor.
 */
ir::Tensor BatchNorm_NCHW(const ir::Tensor &input,
                          const ir::Tensor &weights,
                          float epsilon,
                          const std::string &output_name) {
  CHECK_EQ(4, input->shape.size()) << "Input's dimension of BatchNorm op is not 4! Please check.";
  CHECK_EQ(2, weights->shape.size()) << "Weight's dimension of BatchNorm op is not 2! Please check.";
  auto res = Compute(
      input->shape,
      [=](Expr n, Expr c, Expr h, Expr w) {
        return (((input(n, c, h, w) - weights(Expr(0), c)) / ir::Sqrt(weights(Expr(1), c) + Expr(epsilon))) *
                    weights(Expr(2), c) +
                weights(Expr(3), c));
      },
      output_name);
  return res;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

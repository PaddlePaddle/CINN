#include "cinn/hlir/pe/nn.h"

#include <string>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/context.h"
#include "cinn/common/ir_util.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using cinn::lang::Compute;
using ir::Max;
using ir::Min;
using ir::Select;
using ir::Tensor;

Tensor LeakyRelu(const Tensor &A, double alpha, const std::string &output_name) {
  return Compute(
      A->shape, [=](const std::vector<Expr> &indice) { return LeakyRelu(A(indice), alpha); }, output_name);
}

Tensor PRelu(const Tensor &A, const Tensor &slope, const int axis, const std::string &output_name) {
  CHECK_LT(axis, A->shape.size()) << "Wrong axis value: " << axis << std::endl;
  CHECK(A->shape[axis] == slope->shape[0]) << "Wrong slope shape: " << slope->shape[0] << std::endl;
  return Compute(
      A->shape,
      [=](const std::vector<Expr> &indice) { return LeakyRelu(A(indice), slope(indice[axis])); },
      output_name);
}

std::vector<ir::Tensor> Conv2d_NCHW(const ir::Tensor &input,
                                    const ir::Tensor &weights,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    const std::string &output_name) {
  CHECK_EQ(input->shape.size(), 4U) << "Input's dimension of Conv2d_NCHW op is not 4! Please check.";
  CHECK_EQ(weights->shape.size(), 4U) << "Weight's dimension of Conv2d_NCHW op is not 4! Please check.";
  std::vector<Expr> output_shape;
  std::vector<Expr> new_weights_shape;
  std::vector<Expr> input_pad_shape;
  output_shape = {
      input->shape[0],                                                                                  // B
      weights->shape[0],                                                                                // O
      Expr((input->shape[2] - ((weights->shape[2] - 1) * dilation_h + 1) + 2 * pad_h) / stride_h + 1),  // H
      Expr((input->shape[3] - ((weights->shape[3] - 1) * dilation_w + 1) + 2 * pad_w) / stride_w + 1)   // W
  };
  new_weights_shape = {weights->shape[0],
                       weights->shape[1],
                       dilation_h * (weights->shape[2] - 1) + 1,
                       dilation_w * (weights->shape[3] - 1) + 1};
  input_pad_shape   = {input->shape[0], input->shape[1], input->shape[2] + 2 * pad_h, input->shape[3] + 2 * pad_w};

  auto input_pad = Compute(
      input_pad_shape,
      [=](Expr nn, Expr cc, Expr yy, Expr xx) {
        auto cond =
            ir::logic_and({yy >= pad_h, yy - pad_h < input->shape[2], xx >= pad_w, xx - pad_w < input->shape[3]});
        return ir::Select::Make(cond, input(nn, cc, yy - pad_h, xx - pad_w), ir::Zero(input->type()));
      },
      UniqName("input_pad"));
  auto weights_dilation = Compute(
      new_weights_shape,
      [=](Expr nn, Expr cc, Expr yy, Expr xx) {
        auto cond = ir::logic_and({(xx) % dilation_h == 0, yy % dilation_w == 0});
        return ir::Select::Make(
            cond, weights(nn, cc, yy / dilation_h, xx / dilation_w), common::make_const(weights->type(), 0));
      },
      UniqName("weights_dilation"));

  Var fc(weights->shape[1], UniqName("fc"));
  Var fy(weights_dilation->shape[2], UniqName("fy"));
  Var fx(weights_dilation->shape[3], UniqName("fx"));

  CHECK(MathEqual((weights->shape[0] * weights->shape[1]) % input->shape[1], Expr(0)))
      << "filter's output channel size must be divisible by group\n";
  auto res =
      Compute(output_shape,
              [=](Expr nn, Expr ff, Expr yy, Expr xx) {
                return ir::ReduceSum(
                    input_pad(nn,
                              ff / (weights->shape[0] * weights->shape[1] / input->shape[1]) * weights->shape[1] + fc,
                              yy * stride_h + fy,
                              xx * stride_w + fx) *
                        weights_dilation(ff, fc, fy, fx),
                    common::make_const(input->type(), 0));
              },
              output_name,
              {fc, fy, fx});
  return {input_pad, weights_dilation, res};
}

std::vector<ir::Tensor> Conv2d_NHWC(const ir::Tensor &input,
                                    const ir::Tensor &weights,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    const std::string &output_name) {
  CHECK_EQ(input->shape.size(), 4U) << "Input's dimension of Conv2d_NHWC op is not 4! Please check.";
  CHECK_EQ(weights->shape.size(), 4U) << "Weight's dimension of Conv2d_NHWC op is not 4! Please check.";
  std::vector<Expr> output_shape;
  std::vector<Expr> new_weights_shape;
  std::vector<Expr> input_pad_shape;

  output_shape = {
      input->shape[0],                                                                                  // B
      Expr((input->shape[1] - ((weights->shape[2] - 1) * dilation_h + 1) + 2 * pad_h) / stride_h + 1),  // H
      Expr((input->shape[2] - ((weights->shape[3] - 1) * dilation_w + 1) + 2 * pad_w) / stride_w + 1),  // W
      weights->shape[0]                                                                                 // O
  };
  new_weights_shape = {weights->shape[0],
                       weights->shape[1],
                       dilation_h * (weights->shape[2] - 1) + 1,
                       dilation_w * (weights->shape[3] - 1) + 1};
  input_pad_shape   = {input->shape[0], input->shape[1] + 2 * pad_h, input->shape[2] + 2 * pad_w, input->shape[3]};
  auto input_pad    = Compute(
      input_pad_shape,
      [=](Expr nn, Expr yy, Expr xx, Expr cc) {
        auto cond =
            ir::logic_and({yy >= pad_h, yy - pad_h < input->shape[1], xx >= pad_w, xx - pad_w < input->shape[2]});
        return ir::Select::Make(cond, input(nn, yy - pad_h, xx - pad_w, cc), ir::Zero(input->type()));
      },
      UniqName("input_pad"));

  auto weights_dilation = Compute(
      new_weights_shape,
      [=](Expr nn, Expr cc, Expr yy, Expr xx) {
        auto cond = ir::logic_and({(xx) % dilation_h == 0, yy % dilation_w == 0});
        return ir::Select::Make(
            cond, weights(nn, cc, yy / dilation_h, xx / dilation_w), common::make_const(weights->type(), 0));
      },
      UniqName("weights_dilation"));

  Var fc(weights->shape[1], UniqName("fc"));
  Var fy(weights_dilation->shape[2], UniqName("fy"));
  Var fx(weights_dilation->shape[3], UniqName("fx"));

  CHECK(MathEqual((weights->shape[0] * weights->shape[1]) % input->shape[3], Expr(0)))
      << "filter's output channel size must be divisible by group\n";
  auto res =
      Compute(output_shape,
              [=](Expr nn, Expr yy, Expr xx, Expr ff) {
                return ir::ReduceSum(
                    input_pad(nn,
                              yy * stride_h + fy,
                              xx * stride_w + fx,
                              ff / (weights->shape[0] * weights->shape[1] / input->shape[3]) * weights->shape[1] + fc) *
                        weights_dilation(ff, fc, fy, fx),
                    common::make_const(input->type(), 0));
              },
              output_name,
              {fy, fx, fc});
  return {input_pad, weights_dilation, res};
}

std::vector<Tensor> Depthwise_Conv2d_NCHW(const Tensor &input,
                                          const Tensor &weight,
                                          int pad_h,
                                          int pad_w,
                                          int stride_h,
                                          int stride_w,
                                          const std::string output_name) {
  CHECK_EQ(input->shape.size(), 4U) << "Input's dimension of Depthwise_Conv2d_NCHW is not 4! Please check.\n";
  CHECK_EQ(weight->shape.size(), 4U) << "Weight's dimension of Depthwise_Conv2d_NCHW is not 4! Please check.\n";
  Expr in_h = input->shape[2];
  Expr in_w = input->shape[3];
  Expr c_m  = weight->shape[1];  // channel_multiplier
  std::vector<Expr> output_shape;

  output_shape = {
      input->shape[0],                                                  // B
      weight->shape[1] * input->shape[1],                               // O
      (input->shape[2] - weight->shape[2] + 2 * pad_h) / stride_h + 1,  // H
      (input->shape[3] - weight->shape[3] + 2 * pad_w) / stride_w + 1   // W
  };
  auto input_pad =
      (pad_h == 0 && pad_w == 0) ? Identity(input) : Pad(input, {Expr(0), Expr(0), Expr(pad_h), Expr(pad_w)});

  Var kernel_h = Var(weight->shape[2], "kh");
  Var kernel_w = Var(weight->shape[3], "kw");
  auto res =
      Compute(output_shape,
              [=](Expr nn, Expr ff, Expr yy, Expr xx) {
                return ir::ReduceSum(input_pad(nn, ff / c_m, yy * stride_h + kernel_h, xx * stride_w + kernel_w) *
                                         weight(ff / c_m, ff % c_m, kernel_h, kernel_w),
                                     common::make_const(input->type(), 0));
              },
              output_name,
              {kernel_h, kernel_w});
  return {input_pad, res};
}

std::vector<Tensor> Depthwise_Conv2d_NHWC(const Tensor &input,
                                          const Tensor &weight,
                                          int pad_h,
                                          int pad_w,
                                          int stride_h,
                                          int stride_w,
                                          const std::string output_name) {
  CHECK_EQ(input->shape.size(), 4U) << "Input's dimension of Depthwise_Conv2d_NCHW is not 4! Please check.\n";
  CHECK_EQ(weight->shape.size(), 4U) << "Weight's dimension of Depthwise_Conv2d_NCHW is not 4! Please check.\n";
  Expr in_h = input->shape[1];
  Expr in_w = input->shape[2];
  Expr c_m  = weight->shape[1];  // channel_multiplier
  std::vector<Expr> output_shape;

  output_shape = {
      input->shape[0],                                                  // B
      (input->shape[1] - weight->shape[2] + 2 * pad_h) / stride_h + 1,  // H
      (input->shape[2] - weight->shape[3] + 2 * pad_w) / stride_w + 1,  // W
      weight->shape[1] * input->shape[3]                                // O
  };

  auto input_pad =
      (pad_h == 0 && pad_w == 0) ? Identity(input) : Pad(input, {Expr(0), Expr(pad_h), Expr(pad_w), Expr(0)});

  Var kernel_h = Var(weight->shape[2], "kh");
  Var kernel_w = Var(weight->shape[3], "kw");
  auto res =
      Compute(output_shape,
              [=](Expr nn, Expr yy, Expr xx, Expr ff) {
                return ir::ReduceSum(input_pad(nn, yy * stride_h + kernel_h, xx * stride_w + kernel_w, ff / c_m) *
                                         weight(ff / c_m, ff % c_m, kernel_h, kernel_w),
                                     common::make_const(input->type(), 0));
              },
              output_name,
              {kernel_h, kernel_w});
  return {input_pad, res};
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
                          const ir::Tensor &scale,
                          const ir::Tensor &bias,
                          const ir::Tensor &mean,
                          const ir::Tensor &variance,
                          float epsilon,
                          const std::string &output_name) {
  CHECK_EQ(input->shape.size(), 4U) << "Input's dimension of BatchNorm op is not 4! Please check.";
  CHECK_EQ(scale->shape.size(), 1U) << "Scale's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(bias->shape.size(), 1U) << "Bias's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(mean->shape.size(), 1U) << "Mean's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(variance->shape.size(), 1U) << "Variance's dimension of BatchNorm op is not 1! Please check.";
  auto res = Compute(
      input->shape,
      [=](Expr n, Expr c, Expr h, Expr w) {
        return (input(n, c, h, w) - mean(c)) * scale(c) / Sqrt(variance(c) + Expr(epsilon)) + bias(c);
      },
      UniqName(output_name));
  return res;
}

/**
 * This operator implements the softmax layer.
 * @param A The input tensor.
 * @param axis The axis parameter.
 * @param output_name The name of output tensor.
 * @return The calculated output tensor.
 */
std::vector<ir::Tensor> Softmax(const ir::Tensor &A, int axis, const std::string &output_name) {
  Var axis_j(A->shape[axis], UniqName("axis_j"));
  auto temp      = Compute(A->shape,
                      [=](const std::vector<Expr> &indice) {
                        std::vector<Expr> new_indice = indice;
                        new_indice[axis]             = axis_j;
                        return ir::ReduceSum(Exp(A(new_indice)), Expr(0.f));
                      },
                      UniqName("softmax_temp_out"),
                      {axis_j});
  ir::Tensor out = Compute(
      A->shape,
      [=](const std::vector<Expr> &indice) { return Exp(A(indice)) / temp(indice); },
      UniqName("softmax_out"));
  return {temp, out};
}

ir::Tensor Slice(const ir::Tensor &A,
                 const std::vector<int> &starts,
                 const std::vector<int> &axes,
                 const std::vector<Expr> &output_shape,
                 const std::string &output_name) {
  return Compute(
      output_shape,
      [=](const std::vector<Expr> &indice) {
        std::vector<Expr> temp = indice;
        for (int i = 0; i < axes.size(); i++) {
          temp[axes[i]] = temp[axes[i]] + Expr(starts[i]) + (starts[i] < 0 ? A->shape[axes[i]] : Expr(0));
        }
        return A(temp);
      },
      output_name);
}

/**
 * @brief Perform padding operation.
 * @param tensor The input tensor.
 * @param pad_before Vector of Exprs describing the padding before the respective dimension
 * @param pad_after Vector of Exprs describing the padding after the respective dimension
 * @param pad_value The value to fill padding elements with. Default is zero.
 * @param name The name of the output padding tensor
 * @param pad_mode Padding type to use: "constant" pads with constant_value; "edge" pads using the edge values of the
 * input array; "reflect" pads by reflecting values with respect to the edges.
 *
 * @return the output tensor after padding.
 *
 * @note
 *  The pad_after vector must either be empty or have the same length as pad_before
 *  When pad_after is empty, it takes the same values as pad_before (symmetric padding)
 *  The pad vector applies from the leading dimensions and skips missing trailing dimensions:
 *  e.g.
 *      pad(t(i, j, k), {1}, {1}) returns the equivalent operation for
 *          the following pseudocode:
 *              for i in [0, t.shape[0] + 2):
 *                  for j in [0, t.shape[0] + 2):
 *                      for k in [0, t.shape[0] + 2):
 *                         name(i,j,k) =
 *                             i < 1 ? 0 :
 *                               ((1 <= i < t.shape[0] + 1) ?
 *                                 t(i-1, j, k) : 0));
 *
 */
Tensor Pad(const Tensor &tensor,
           const std::vector<Expr> &pad_before,
           std::vector<Expr> pad_after,
           Expr pad_value,
           const std::string &name,
           const std::string &pad_mode) {
  // When pad_after is empty, it takes the same values as pad_before (symmetric padding)
  if (pad_after.size() < pad_before.size()) {
    for (size_t i = pad_after.size(); i < pad_before.size(); ++i) {
      pad_after.push_back(pad_before[i]);
    }
  }
  CHECK(!pad_before.empty());
  CHECK_EQ(pad_before.size(), pad_after.size());
  std::vector<Expr> output_shape;
  for (auto &ele : pad_before) {
    CHECK(ele.type().is_int(32)) << "padding size should be int32\n";
  }
  for (auto &ele : pad_after) {
    CHECK(ele.type().is_int(32)) << "padding size should be int32\n";
  }
  for (size_t i = 0; i < tensor->shape.size(); ++i) {
    if (i >= pad_before.size()) {
      output_shape.push_back(tensor->shape[i]);
    } else {
      auto shape = common::AutoSimplify(tensor->shape[i] + pad_before[i] + pad_after[i]);
      output_shape.push_back(shape);
    }
  }
  // default value is zero
  if (!pad_value.defined()) {
    pad_value = make_const(tensor->type(), 0);
  }

  auto fn = [=](const std::vector<Expr> &ovars) {
    std::vector<Expr> indices;
    std::vector<Expr> sel;
    std::vector<Expr> pad_idx;
    for (size_t i = 0; i < tensor->shape.size(); ++i) {
      if (i >= pad_before.size()) {
        indices.emplace_back(ovars[i]);
        continue;
      }
      if (!MathEqual(pad_before[i], Expr(0))) {
        sel.push_back(ir::GE::Make(ovars[i], pad_before[i]));
        indices.push_back(ovars[i] - pad_before[i]);
      } else {
        indices.emplace_back(ovars[i]);
      }
      Expr sel_after;
      if (!MathEqual(pad_after[i], Expr(0))) {
        sel_after = common::AutoSimplify(ovars[i] < pad_before[i] + tensor->shape[i]);
        sel.push_back(sel_after);
      }
      if (pad_mode == "edge") {
        pad_idx.push_back(Select::Make(
            ovars[i] < pad_before[i],
            0,
            Select::Make(
                ovars[i] >= pad_before[i] + tensor->shape[i], tensor->shape[i] - 1, ovars[i] - pad_before[i])));
      } else if (pad_mode == "reflect") {
        pad_idx.push_back(Select::Make(ovars[i] < pad_before[i],
                                       pad_before[i] - ovars[i],
                                       Select::Make(ovars[i] >= pad_before[i] + tensor->shape[i],
                                                    tensor->shape[i] * 2 - ovars[i] + pad_before[i] - 2,
                                                    ovars[i] - pad_before[i])));
      }
    }
    if (sel.size() != 0) {
      auto fn = [](Expr a, Expr b) { return a && b; };
      if (pad_mode == "constant") {
        return Select::Make(FoldExpr(fn, sel), tensor(indices), pad_value);
      } else if (pad_mode == "edge" || pad_mode == "reflect") {
        return Select::Make(FoldExpr(fn, sel), tensor(indices), tensor(pad_idx));
      }
    }
    return tensor(indices);
  };
  return Compute(output_shape, fn, UniqName(name));
}

/**
 * @brief Perform pooling on N-dimension of data.
 *
 * @param tensor The input tensor with the shape of {N, C, H, W} or {N, H, W, C}.
 * @param kernel_size Vector of N ints that indicates pooling kernel size. If N is 2, then is {pool_kernel_Height,
 * pool_kernel_Width}.
 * @param stride_size Vector of N ints that indicates pooling stride size. If N is 2, then is {pool_stride_Height,
 * pool_stride_Width}.
 * @param padding_size Vector of N*2 ints {head_pad_d1, head_pad_d2, ..., head_pad_dN, tail_pad_d1, tail_pad_d2, ...,
 * tail_pad_dN}. If N is 2, then is {pad_height_top, pad_width_left, pad_height_bottom, pad_width_right]}.
 * @param pool_type The type of pooling operator, currently support "max" and "avg".
 * @param axis Vector of axes of the tensor for pooling.
 * @param ceil_mode Whether to use ceil when calculating the output size.
 * @param exclusive Whether include padding in the calculation'.
 * @param output_name the name of the output tensor after padding and pooling.
 *
 * @return the vector of padding tensor and pooling tensor
 */
std::vector<Tensor> PoolImpl(const Tensor &tensor,
                             const std::vector<int> &kernel_size,
                             const std::vector<int> &stride_size,
                             const std::vector<int> &padding_size,
                             const std::string &pool_type,
                             const std::vector<int> &axis,
                             bool ceil_mode,
                             bool exclusive,
                             const std::string &output_name) {
  LOG(INFO) << "kernel_size length is: " << kernel_size.size();
  LOG(INFO) << "kernel_size is: " << kernel_size[0];
  LOG(INFO) << "padding_size length is: " << padding_size.size();
  LOG(INFO) << "padding_size is: " << padding_size[0];
  CHECK(!kernel_size.empty()) << "Pooling kernel_size should not be empty\n";
  int k_size = kernel_size.size();
  int x_size = tensor->shape.size();
  CHECK_EQ(stride_size.size(), k_size) << "Pooling stride_size must have same elements as kernel\n";
  CHECK_EQ(padding_size.size(), k_size * 2) << "Pooling padding_size must have double elements as kernel\n";
  CHECK_EQ(axis.size(), k_size) << "Axis must have same elements as kernel\n";

  std::vector<Var> daxis;
  std::vector<Expr> kernel(k_size);
  std::vector<Expr> stride(k_size);
  std::vector<Expr> pad_head(k_size);
  std::vector<Expr> pad_tail(k_size);
  std::vector<Expr> pad_before(x_size, Expr(0));
  std::vector<Expr> pad_after(x_size, Expr(0));
  std::vector<Expr> out_shape = tensor->shape;

  bool do_pad = false;
  for (int i = 0; i < k_size; i++) {
    int ii      = axis[i];
    kernel[i]   = Expr(kernel_size[i]);
    stride[i]   = Expr(stride_size[i]);
    pad_head[i] = Expr(padding_size[i]);
    pad_tail[i] = Expr(padding_size[i + k_size]);
    do_pad      = (do_pad) ? do_pad : (padding_size[i] || padding_size[i + k_size]);

    if (ceil_mode) {
      pad_tail[i] = common::AutoSimplify(pad_tail[i] + stride[i] - 1);
    }

    daxis.emplace_back(Var(kernel[i], UniqName("kernel_idx")));

    pad_before[ii] = pad_head[i];
    pad_after[ii]  = pad_tail[i];

    auto out_dim = common::AutoSimplify((tensor->shape[ii] - kernel[i] + pad_head[i] + pad_tail[i]) / stride[i] + 1);

    out_shape[ii] = out_dim;
  }

  Tensor temp;
  Tensor res;
  if (pool_type == "max") {
    Expr min_value = ir::min_value(tensor->type());
    // Pad the input tensor with the pad_value of type's minimum value
    temp = do_pad ? Pad(tensor, pad_before, pad_after, min_value, UniqName("pad_temp")) : tensor;
    res  = Compute(
        out_shape,
        [=](const std::vector<Expr> &output) {
          std::vector<Expr> indices;
          for (auto &var : output) indices.push_back(var);

          for (int i = 0; i < k_size; i++) {
            int ii      = axis[i];
            indices[ii] = output[ii] * stride[i] + daxis[i];
          }

          return ReduceMax(temp(indices), min_value);
        },
        UniqName(output_name),
        daxis);
  } else if (pool_type == "avg") {
    // Pad the input tensor with pad_value zero
    temp = do_pad ? Pad(tensor, pad_before, pad_after, 0, UniqName("pad_temp")) : tensor;
    res  = Compute(
        out_shape,
        [=](const std::vector<Expr> &output) {
          std::vector<Expr> indices;
          for (const Expr &var : output) indices.push_back(var);

          for (int i = 0; i < k_size; i++) {
            int ii      = axis[i];
            indices[ii] = output[ii] * stride[i] + daxis[i];
          }

          if (exclusive) {
            std::vector<Expr> start(k_size);
            std::vector<Expr> end(k_size);
            auto kernel_size = make_const(Int(32), 1);
            for (int i = 0; i < k_size; i++) {
              int ii      = axis[i];
              start[i]    = common::AutoSimplify(output[ii] * stride[i] - pad_head[i]);
              end[i]      = Min::Make(start[i] + kernel[i], tensor->shape[ii]);
              start[i]    = Max::Make(start[i], make_const(Int(32), 0));
              kernel_size = kernel_size * (end[i] - start[i]);
            }
            common::AutoSimplify(kernel_size);
            Expr divide_factor = Max::Make(kernel_size, make_const(Int(32), 1));
            return ReduceSum(ir::Div::Make(temp(indices), cast(divide_factor, Float(32))), Expr());
          } else {
            auto kernel_size = make_const(Int(32), 1);
            for (int i = 0; i < k_size; i++) {
              kernel_size = kernel_size * kernel[i];
            }
            common::AutoSimplify(kernel_size);
            return ReduceSum(ir::Div::Make(temp(indices), cast(kernel_size, Float(32))), Expr());
          }
        },
        UniqName(output_name),
        daxis);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
  }
  if (do_pad) {
    return {temp, res};
  } else {
    return {res};
  }
}

std::vector<Tensor> Pool1d(const Tensor &tensor,
                           const std::vector<int> &kernel_size,
                           const std::vector<int> &stride_size,
                           const std::vector<int> &padding_size,
                           const std::string &pool_type,
                           bool ceil_mode,
                           bool exclusive,
                           const std::string &data_format,
                           const std::string &output_name) {
  int width_axis = -1;
  if (data_format == "NCW") {
    width_axis = 2;
  } else if (data_format == "NWC") {
    width_axis = 1;
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format << std::endl;
  }
  CHECK_EQ(tensor->shape.size(), 3U) << "pool1d requires tensor's shape_size to be 3\n";
  std::vector<int> axis = {width_axis};
  return PoolImpl(
      tensor, kernel_size, stride_size, padding_size, pool_type, axis, ceil_mode, exclusive, UniqName(output_name));
}

std::vector<Tensor> Pool2d(const Tensor &tensor,
                           const std::vector<int> &kernel_size,
                           const std::vector<int> &stride_size,
                           const std::vector<int> &padding_size,
                           const std::string &pool_type,
                           bool ceil_mode,
                           bool exclusive,
                           const std::string &data_format,
                           const std::string &output_name) {
  int height_axis = -1;
  int width_axis  = -1;
  if (data_format == "NCHW") {
    height_axis = 2;
    width_axis  = 3;
  } else if (data_format == "NHWC") {
    height_axis = 1;
    width_axis  = 2;
  } else if (data_format == "AnyLayout") {
    height_axis = 2;
    width_axis  = 3;
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format << std::endl;
  }
  CHECK_EQ(tensor->shape.size(), 4U) << "pool1d requires tensor's shape_size to be 4\n";
  std::vector<int> axis = {height_axis, width_axis};
  return PoolImpl(
      tensor, kernel_size, stride_size, padding_size, pool_type, axis, ceil_mode, exclusive, UniqName(output_name));
}

std::vector<Tensor> Pool3d(const Tensor &tensor,
                           const std::vector<int> &kernel_size,
                           const std::vector<int> &stride_size,
                           const std::vector<int> &padding_size,
                           const std::string &pool_type,
                           bool ceil_mode,
                           bool exclusive,
                           const std::string &data_format,
                           const std::string &output_name) {
  int height_axis = -1;
  int width_axis  = -1;
  int depth_axis  = -1;
  if (data_format == "NCDHW") {
    depth_axis  = 2;
    height_axis = 3;
    width_axis  = 4;
  } else if (data_format == "NDHWC") {
    depth_axis  = 1;
    height_axis = 2;
    width_axis  = 3;
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format << std::endl;
  }
  CHECK_EQ(tensor->shape.size(), 5U) << "pool1d requires tensor's shape_size to be 5\n";
  std::vector<int> axis = {depth_axis, height_axis, width_axis};
  return PoolImpl(
      tensor, kernel_size, stride_size, padding_size, pool_type, axis, ceil_mode, exclusive, UniqName(output_name));
}

Tensor DropoutInfer(const ir::Tensor &tensor,
                    float dropout_prob,
                    const std::string &dropout_implementation,
                    const std::string &output_name) {
  if (dropout_implementation == "downgrade_in_infer") {
    return Multiply(tensor, Expr(1 - dropout_prob));
  } else if (dropout_implementation == "upscale_in_train") {
    return Identity(tensor);
  } else {
    LOG(FATAL) << "dropout_implementation attr must be 'downgrade_in_infer' or 'upscale_in_train'\n";
  }
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

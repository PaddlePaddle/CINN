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

#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "cinn/common/macros.h"
#include "cinn/common/type.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/utils/functional.h"
#include "cinn/utils/type_defs.h"

namespace cinn {
namespace frontend {

// clang-format off

// ******************************************* //
// Elementwise compute each element in `input` variable, and return the result Variable.
// Variable UNARY_OP(const Variable& x);
#define NETBUILDER_UNARY_OP_FOREACH(macro__) \
  macro__(Sqrt) \
  macro__(Tanh) \
  macro__(Relu) \
  macro__(Sigmoid) \
  macro__(Identity) \
  macro__(Exp) \
  macro__(Erf) \
  macro__(Rsqrt) \
  macro__(Log) \
  macro__(Log2) \
  macro__(Log10) \
  macro__(Floor) \
  macro__(Ceil) \
  macro__(Round) \
  macro__(Trunc) \
  macro__(Sin) \
  macro__(Cos) \
  macro__(Tan) \
  macro__(Sinh) \
  macro__(Cosh) \
  macro__(Asin) \
  macro__(Acos) \
  macro__(Atan) \
  macro__(Asinh) \
  macro__(Acosh) \
  macro__(Atanh) \
  macro__(IsNan) \
  macro__(IsFinite) \
  macro__(IsInf) \
  macro__(LogicalNot) \
  macro__(BitwiseNot) \
  macro__(Negative) \
  macro__(Sign) \
  macro__(Abs)

// ******************************************* //
// The op has two input and one output, with a attribute [axis]
// Variable BINARY_OP(const Variable& lhs, const Variable& rhs, int axis = -1);
#define NETBUILDER_BINARY_OP_FOREACH(macro__) \
  macro__(Add) \
  macro__(Subtract) \
  macro__(Divide) \
  macro__(Multiply) \
  macro__(FloorDivide) \
  macro__(Mod) \
  macro__(FloorMod) \
  macro__(Max) \
  macro__(Min) \
  macro__(Power) \
  macro__(LogicalAnd) \
  macro__(LogicalOr) \
  macro__(LogicalXor) \
  macro__(BitwiseAnd) \
  macro__(BitwiseOr) \
  macro__(BitwiseXor) \
  macro__(LeftShift) \
  macro__(RightShift) \
  macro__(Equal) \
  macro__(NotEqual) \
  macro__(GreaterThan) \
  macro__(LessThan) \
  macro__(GreaterEqual) \
  macro__(LessEqual)

// ******************************************* //
// Reduce array elements over the given dims.
// Variable REDUCE_OP(const Variable& x, const std::vector<int>& dim = {}, bool keep_dim = false);
#define NETBUILDER_REDUCE_OP_FOREACH(macro__) \
  macro__(ReduceSum) \
  macro__(ReduceProd) \
  macro__(ReduceMax) \
  macro__(ReduceMin) \
  macro__(ReduceAll) \
  macro__(ReduceAny)
// clang-format on

class NetBuilder {
  using AttributeMap = utils::AttributeMap;

 private:
  std::string name_;
  std::vector<Instruction> instrs_;
  std::vector<Variable> inputs_;

 public:
  // class base API
  explicit NetBuilder(const std::string& name);

  Program Build(bool in_reverse = false);

  // name of this builder
  const std::string& name() { return name_; }

  // the number of instructions
  const size_t size() { return instrs_.size(); }

  virtual ~NetBuilder() = default;

  void AppendInstruction(const Instruction& instr) { instrs_.push_back(instr); }

  void InferShape(Instruction instr) const;

  const std::vector<Variable>& CustomInstr(const std::string& type,
                                           const std::vector<Variable>& inputs,
                                           const AttributeMap& attrs);

 protected:
  /**
   * @brief Helper function of UnaryOp.
   *
   * @param op_type The unary op name.
   * @param x The input variable.
   *
   * @return The result variable.
   */
  Variable UnaryOp(const std::string& op_type, const Variable& x);

  /**
   * @briefHelper function of BinaryOp.
   *
   * @param op_type The binary op name.
   * @param lhs The left input variable.
   * @param rhs The right input variable.
   * @param axis The compute axis. Default is -1.
   *
   * @return The result variable.
   */
  Variable BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs, int axis = -1);

  /**
   * @brief Reduce array elements over the given dims.
   *
   * @param op_type The reduce op name.
   * @param x The input variable.
   * @param dim The dims along which a sum is performed. If dim is empty, the operation will sum over all elements
   * of the input array. If the dim has negative value, it should count from the last dim to the first. Default is None.
   * @param keep_dim If it is set true, the axes which are reduced are left in the result as dimensions with size one.
   * With this option, the result will broadcast correctly against the input array. Default is false.
   *
   * @return The result variable.
   */
  Variable Reduce(const std::string& op_type,
                  const Variable& x,
                  const std::vector<int>& dim = {},
                  bool keep_dim               = false);

 private:
  // the helper function for Matmul API
  std::pair<Variable, Variable> BroadcastMatmulInput(
      const Variable& x, const Variable& y, bool trans_x, bool trans_y, float alpha);
  std::vector<int> GetMatmulOutputShape(const Variable& x, const Variable& y, bool trans_x, bool trans_y, float alpha);

 public:
  // *******************************************
  // Elementwise Operator
  /**
   * @brief Elementwise compute each element in `input` variable, and return the result Variable.
   * @param x The input variable.
   * @return The output variable.
   */
#define NETBUILDER_UNARY_OP_DECL(func_name__) Variable func_name__(const Variable& x);
  NETBUILDER_UNARY_OP_FOREACH(NETBUILDER_UNARY_OP_DECL)
#undef NETBUILDER_UNARY_OP_DECL

  /**
   * @brief Compute each each element in `lhs` variable and `rhs` variable in `axis` dimension, and return the result
   * Variable.
   * @param lhs The left input variable.
   * @param rhs The right input variable.
   * @param axis The compute axis. Default is -1.
   * @return The result variable.
   */
#define NETBUILDER_BINARY_OP_DECL(func_name__) \
  Variable func_name__(const Variable& lhs, const Variable& rhs, int axis = -1);
  NETBUILDER_BINARY_OP_FOREACH(NETBUILDER_BINARY_OP_DECL)
#undef NETBUILDER_BINARY_OP_DECL

  /**
   * @brief Return array elements depending on condition.
   * @param condition The condition which determine return `true_value` or `false_value`.
   * @param true_value Return `true_value` if the element of `condition` is true.
   * @param false_value Return `false_value` if the element of `condition` is false.
   * @return The result variable.
   */
  Variable Select(const Variable& condition, const Variable& true_value, const Variable& false_value);

  /**
   * @brief Scale operator.
   * @param x Input N-D variable of scale operator.
   * @param scale The scale factor of the input. Default is 1.0f.
   * @param bias The bias to be put on the input. Default is 0.0f.
   * @param bias_after_scale Apply bias addition after or before scaling. It is useful for numeric stability in some
   * circumstances. Default is true.
   * @return Output variable of scale operator, with shape and data type same as input.
   */
  Variable Scale(const Variable& x, float scale = 1.0f, float bias = 0.0f, bool bias_after_scale = true);

  /**
   * @brief This OP is used to sum one or more variable of the input.
   * @param x A Varaible list. The shape and data type of the list elements should be consistent.
   * @return The sum of input `x`. its shape and data types are consistent with `x`.
   */
  Variable Sum(const std::vector<Variable>& inputs);

  /**
   * @brief Drop or keep each element of x independently.
   * @param x The input variable.
   * @param dropout_prob Probability of setting units to zero. The dropout operator randomly sets (according to the
   * given dropout probability) the outputs of some units to zero, while others are remain unchanged.
   * @param dropout_implementation Choice the mode of dropout. When "downgrade_in_infer", downgrade the outcome at
   * inference: `train: out = input * mask, inference: out = input * (1.0 - dropout_prob)`. When "upscale_in_train",
   * upscale the outcome at training time: `train: out = input * mask / ( 1.0 - dropout_prob), inference: out = input`.
   * @return A variable representing the dropout, has same shape and data type with input.
   */
  Variable DropoutInfer(const Variable& x,
                        float dropout_prob                        = 0.5f,
                        const std::string& dropout_implementation = "downgrade_in_infer");

  /**
   * @brief The clip operator limits the value of given input within an interval `[min, max]`.
   * @param x Input N-D variable of scale operator.
   * @param max The minimum value to clip by.
   * @param min The maximum value to clip by
   * @return Output of clip with the same shape and data type as input.
   */
  Variable Clip(const std::vector<Variable>& x, const float& max, const float& min);

  /**
   * @brief This operator checks if all `x` and `y` satisfy the condition: `|x - y| <= atol + rtol * |y|`
   * @param x The first variable.
   * @param y The second variable.
   * @param rtol The relative tolerance. Default: 1e−5f.
   * @param atol The absolute tolerance. Default: 1e−8f.
   * @param equal_nan If `true`, then two NaNs will be compared as equal. Default: false .
   * @return The output variable, it’s data type is bool.
   */
  Variable IsClose(
      const Variable& x, const Variable& y, float rtol = 1e-05f, float atol = 1e-08f, bool equal_nan = false);

  // *******************************************
  // Reduction operator
  /**
   * @brief Reduce array elements over the given dims.
   * @param x The input variable.
   * @param dim The dims along which a sum is performed. If dim is empty, the operation will sum over all elements
   * of the input array. If the dim has negative value, it should count from the last dim to the first. Default is None.
   * @param keep_dim If it is set true, the axes which are reduced are left in the result as dimensions with size one.
   * With this option, the result will broadcast correctly against the input array. Default is false.
   * @return The result variable.
   */
#define NETBUILDER_REDUCE_OP_DECL(func_name__) \
  Variable func_name__(const Variable& x, const std::vector<int>& dim = {}, bool keep_dim = false);
  NETBUILDER_REDUCE_OP_FOREACH(NETBUILDER_REDUCE_OP_DECL)
#undef NETBUILDER_REDUCE_OP_DECL

  // *******************************************
  // Tensor operator
  /**
   * @brief Copy and create input from `input` variable.
   * @param input The input variable.
   * @return The new input.
   */
  Placeholder CreateInput(const Variable& input);

  /**
   * @brief Create new input, whose data type is `type`, shape is `shape`, and id is `id_hint`.
   * @param type The input variable's data type.
   * @param shape The input variable's shape.
   * @param id_hint The input variable's name. Default is None.
   * @return The new input.
   */
  Placeholder CreateInput(const common::Type& type, const std::vector<int>& shape, const std::string& id_hint = "");

  /**
   * @brief Create constant tensor with the specific value/vector and type, the type is infered from value.
   * @param value The constant value to be set.
   * @param name The name of output variable.
   * @return The result variable.
   */
  template <typename T>
  std::enable_if_t<std::is_fundamental<T>::value, Variable> Constant(const T& value, const std::string& name) {
    Instruction instr("const_scalar");
    instr.SetInputs({});
    instr.SetAttr<T>("value", value);

    InferShape(instr);
    AppendInstruction(instr);

    auto out = instr.GetOutput(0);
    out.set_id(name);
    out->type = common::type_of<T>();
    return out;
  }

  template <typename T>
  std::enable_if_t<cinn::utils::is_vector<T>::value, Variable> Constant(const T& value, const std::string& name) {
    CHECK(!value.empty()) << "The value of Constant should not be None or empty list! Please check.";
    if (value.size() == 1UL) {
      return Constant(value.front(), cinn::UniqName(name));
    }

    std::vector<Variable> tmp;
    for (int i = 0; i < value.size(); ++i) {
      auto var = Constant(value[i], cinn::UniqName(name + "_" + std::to_string(i)));
      if (var->shape == std::vector<int>{1}) {
        tmp.emplace_back(var);
      } else {
        auto new_shape = var->shape;
        new_shape.insert(new_shape.begin(), 1);
        tmp.emplace_back(Reshape(var, new_shape));
      }
    }
    auto out = Concat(tmp);
    out.set_id(name);
    return out;
  }

  /**
   * @brief The op return a variable with the specific value, shape and type.
   * @param shape Shape of the variable to be created.
   * @param value The constant value used to initialize the variable to be created.
   * @param name The name of the output variable.
   * @param dtype Data type of the output variable.
   * @param force_cpu Whether the variable should force placed in cpu, default in device memory. Default is false.
   * @return The result variable.
   */
  Variable FillConstant(const std::vector<int>& shape,
                        float value,
                        const std::string& name,
                        const std::string& dtype,
                        bool force_cpu = false);

  /**
   * @brief The op return a variable with the specific value, shape and type, the type is infered from value.
   * @param shape Shape of the variable to be created.
   * @param value The constant value used to initialize the variable to be created.
   * @param name The name of the output variable.
   * @param force_cpu Whether the variable should force placed in cpu, default in device memory. Default is false.
   * @return The result variable.
   */
  template <typename T = float>
  Variable FillConstant(const std::vector<int>& shape, T value, const std::string& name, bool force_cpu = false) {
    return FillConstant(shape, static_cast<float>(value), name, common::Type2Str(common::type_of<T>()), force_cpu);
  }

  /**
   * @brief Return evenly spaced values within a given interval. Values are generated within the half-open interval
   * `[start, stop)` (in other words, the interval including start but excluding stop).
   * @param start Start of interval. The interval includes this value.
   * @param stop End of interval. The interval does not include this value, except in some cases where step is not
   * an integer and floating point round-off affects the length of out.
   * @param step Spacing between values. For any output out, this is the distance between two adjacent values, `out[i+1]
   * - out[i]`.
   * @param dtype The data type of the output. Default: "float32".
   * @return A 1-D variable which is evenly spaced values within a given interval. Its data type is set by dtype.
   */
  Variable Arange(const float start, const float stop, const float step, const std::string& dtype);

  /**
   * @brief This operator is used to perform matrix multiplication for input x and y.
   * @param x The first input variable.
   * @param y The second input variable.
   * @param x_num_col_dims If the input `x` is a variable with more than two dimensions, `x` will be flattened into a
   * two-dimensional matrix first. The flattening rule is: the first `num_col_dims` will be flattened to form the first
   * dimension of the final matrix (the height of the matrix), and the rest `rank(x)` - `num_col_dims` dimensions are
   * flattened to form the second dimension of the final matrix (the width of the matrix). Default is 1.
   * @param y_num_col_dims If the input `y` is a variable with more than two dimensions, `y` will be flattened into a
   * two-dimensional matrix first. The attribute `y_num_col_dims` determines how `y` is flattened. See comments of
   * `x_num_col_dims` for more details. Default is 1.
   * @return The result variable.
   */
  Variable Mul(const Variable& x, const Variable& y, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * @brief Applies matrix multiplication to two variable. Matmul follows the complete broadcast rules, and its behavior
   * is consistent with `np.matmul`.
   * @param x The left input variable.
   * @param y The right input variable.
   * @param trans_x Whether to transpose `x` before multiplication. Default is false.
   * @param trans_y Whether to transpose `y` before multiplication. Default is false.
   * @param alpha The scale of output. Default 1.0f.
   * @return The product variable.
   */
  Variable Matmul(const Variable& x, const Variable& y, bool trans_x = false, bool trans_y = false, float alpha = 1.0f);

  /**
   * @brief This operation calculates the pooling output based on the input, pooling_type and pool_size, pool_stride,
   * pool_padding parameters.
   * @param x The input variable of pooling operator which is a 4-D variable with shape [N, C, H, W]. The format of
   * input variable is “NCHW” or “NHWC”, where N is batch size, C is the number of channels, H is the height of the
   * feature, and W is the width of the feature.
   * @param pooling_type pooling type, can be “max” for max-pooling and “avg” for average-pooling
   * @param ksize The pool kernel size. If pool kernel size is a tuple or list, it must contain two integers,
   * (pool_size_Height, pool_size_Width). Otherwise, the pool kernel size will be a square of an int.
   * @param strides  The pool stride size. If pool stride size is a tuple or list, it must contain two integers,
   * (pool_stride_Height, pool_stride_Width). Otherwise, the pool stride size will be a square of an int. Default is {1,
   * 1}.
   * @param paddings he padding size. If padding is a list/tuple, it must contain two integers, (padding_H, padding_W).
   * Otherwise, the padding_H = padding_W = padding. Default: padding = {0, 0}.
   * @param ceil_mode Whether to use the ceil function to calculate output height and width. False is the default. If it
   * is set to False, the floor function will be used. Default False
   * @param exclusive Whether to exclude padding points in average pooling mode, default is true.
   * @param global_pooling Whether to use the global pooling. If global_pooling = true, kernel size and paddings will be
   * ignored. Default False
   * @param data_format Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
   * @param adaptive When true, will perform adaptive pooling instead, output shape in H and W dimensions will be same
   * as ksize, input data will be divided into grids specify by ksize averagely and perform pooling in each grid area to
   * get output pooling value. Default: False.
   * @param padding_algorithm CINN not support! It can be "EXPLICIT"/"SAME"/"VALID". Default: "EXPLICIT".
   * @return The output variable of pooling result. The data type is same as input variable.
   */
  Variable Pool2d(const Variable& x,
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

  // *******************************************
  // Broadcast operator
  /**
   * @brief Broadcast the input variable to a given shape.
   * @param x The input variable need to broadcast.
   * @param out_shape The result shape after broadcasting. The value -1 in shape means keeping the corresponding
   * dimension unchanged.
   * @return The result variable with given shape.
   */
  Variable BroadcastTo(const Variable& x, const std::vector<int>& out_shape);

  /**
   * @brief Broadcast the input variable to a given shape.
   * @param x The input variable need to broadcast.
   * @param out_shape  The result shape after broadcasting. The value -1 in shape means keeping the corresponding
   * dimension unchanged.
   * @param broadcast_axes The axes need to broadcast, the axis not in `broadcast_axes` of `out_shape`'s value should be
   * the same as input shape.
   * @return The result variable with given shape.
   */
  Variable BroadcastTo(const Variable& x, const std::vector<int>& out_shape, const std::vector<int>& broadcast_axes);

  // *******************************************
  // Data Layout transform operator
  /**
   * @brief This OP concatenates the input along the axis.
   * @param x Variable list with same data type.
   * @param axis Specify the axis to operate on the input concatenates. The effective range is [-R, R), where R is
   * Rank(x). When axis < 0, it works the same way as axis+R. Default is 0.
   * @return A variable with the same data type as x.
   */
  Variable Concat(const std::vector<Variable>& x, int axis = 0);

  /**
   * @brief Split the input variable into multiple sub-variables.
   * @param x A N-D variable.
   * @param num_or_sections If `num_or_sections` is an int, then `num_or_sections` indicates the number of equal sized
   * sub-variables that the `x` will be divided into. If `num_or_sections` is a list, the length of it indicates the
   * number of sub-variables and the elements in it indicate the sizes of sub-variables’ dimension orderly. The length
   * of the list must not be larger than the `x`'s size of specified axis.
   * @param axis The axis along which to split. The effective range is [-R, R), where R is Rank(x). When axis < 0, it
   * works the same way as axis+R. Default is 0.
   * @return The list of segmented variables.
   */
  std::vector<Variable> Split(const Variable& x, const std::vector<int>& num_or_sections, int axis = 0);

  /**
   * @brief This operator changes the shape of x without changing its data.
   * @param x An N-D variable.
   * @param shape Define the target shape. At most one dimension of the target shape can be -1.
   * @return A reshaped variable with the same data type as x.
   */
  Variable Reshape(const Variable& x, const std::vector<int>& shape);

  /**
   * @brief This OP will squeeze single-dimensional entries of input variable shape. If axes is provided, will remove
   * the dims by axes, the dims selected by axes should be one. If not provide axes, all dims equal to one will be
   * deleted.
   * @param x An N-D variable.
   * @param axes The dimensions to be squeezed. Axes range is `[−rank(input),rank(input)]`. If `axes` is negative,
   * `axes=axes+rank(input)`.
   * @return Output squeezed variable. Data type is same as input variable.
   */
  Variable Squeeze(const Variable& x, const std::vector<int>& axes);

  /**
   * @brief This operator reverse the input along the axis.
   * @param x An N-D variable.
   * @param axis Specify the axis to operate on the input reverse.
   * @return A reversed variable with the same data type as x.
   */
  Variable Reverse(const Variable& x, const std::vector<int>& axis);

  /**
   * @brief Permute the data dimensions of input according to perm. The i-th dimension of the returned variable will
   * correspond to the perm[i]-th dimension of input.
   * @param x An N-D variable.
   * @param axis Permute the input according to the data of perm.
   * @return A transposed n-D variable.
   */
  Variable Transpose(const Variable& x, const std::vector<int>& axis);

  /**
   * @brief This operator produces a slice of x along multiple axes.
   * @param x An N-D variable.
   * @param axes Axes that starts and ends apply to.
   * @param starts The starting indices of corresponding axis in axes. Default: None.
   * @param ends The ending indices of corresponding axis in axes. Default: None.
   * @param infer_flags Whether the output shape can be infered in compile time. Now only support all 1. Default: None.
   * @param strides The slice step of corresponding axis in axes. Default: None.
   * @return A variable with the same dimension as x. The data type is same as x.
   */
  Variable Slice(const Variable& x,
                 const std::vector<int>& axes,
                 const std::vector<int>& starts      = {},
                 const std::vector<int>& ends        = {},
                 const std::vector<int>& infer_flags = {},
                 const std::vector<int>& strides     = {});

  /**
   * @brief Returns a new variable which indexes the input variable along dimension axis using the entries in index
   * which is a variable. The returned variable has the same number of dimensions as the original x variable. The dim-th
   * dimension has the same size as the length of index; other dimensions have the same size as in the x variable.
   * @param x An N-D variable.
   * @param index The 1-D variable containing the indices to index.
   * @param axis  The dimension in which we index. Default: 0.
   * @return A variable with same data type as x.
   */
  Variable IndexSelect(const Variable& x, const Variable& index, int axis = 0);

  /**
   * @brief Output is obtained by updating the input on selected indices based on updates.
   * @param x  The input N-D variable with ndim>=1.
   * @param updates pdate input with updates parameter based on index. shape should be the same as input, and dim value
   * with dim > 1 should be the same as input.
   * @param index The index 1-D variable. The length of index cannot exceed updates’s length, and the value in index
   * cannot exceed input’s length.
   * @param axis  The dimension in which we index. Default: 0.
   * @return A variable with same shape as x.
   */
  Variable ScatterAssign(const Variable& x, const Variable& updates, const Variable& index, int axis = 0);

  /**
   * @brief Output is obtained by adding the `input` and the `updates` on selected indices.
   * @param x  The input N-D variable with ndim>=1.
   * @param updates Update input with updates parameter based on index. Shape should be the same as input, and dim value
   * with dim > 1 should be the same as input.
   * @param index The index 1-D variable. The length of index cannot exceed updates’s length, and the value in index
   * cannot exceed input’s length.
   * @param axis  The dimension in which we index. Default: 0.
   * @return A variable with same shape as x.
   */
  Variable ScatterAdd(const Variable& x, const Variable& updates, const Variable& index, int axis = 0);

  /**
   * @brief Replacing the value of `x` by `assign` variable on the range of `slice(x)`. In other word,
   * `slice(x)=assign`.
   * @param x An N-D variable.
   * @param assign Update input with assign value based on slice result. Shape should be the same as the `slice` output
   * shape.
   * @param axes Axes that starts and ends apply to.
   * @param starts The starting indices of corresponding axis in axes.
   * @param ends The ending indices of corresponding axis in axes.
   * @param strides The slice step of corresponding axis in axes.Default: None.
   * @return A variable with the same shape as x. The data type is same as x.
   */
  Variable SliceAssign(const Variable& x,
                       const Variable& assign,
                       const std::vector<int>& axes,
                       const std::vector<int>& starts,
                       const std::vector<int>& ends,
                       const std::vector<int>& strides = {});

  // *******************************************
  // Activation Operator
  /**
   * @brief Relu6 Activation Operator.
   * @param x Input of relu6 operator, an N-D Tensor.
   * @param threshold  The threshold value of Relu6. Default is 6.0f.
   * @return Output of relu6 operator, a variable with the same shape as input.
   */
  Variable Relu6(const Variable& x, float threshold = 6.0f);

  /**
   * @brief This operator implements the softmax layer.
   * @param x An N-D variable.
   * @param axis The index of dimension to perform softmax calculations, it should be in range `[−1,rank−1]`,
   * while `rank` is the rank of input variable. Default: -1. -1 means the last dimension.
   * @param data_format Specify the data format of the output data, the input will be transformed automatically.
   * An optional string from: "AnyLayout", "NHWC", "NCHW". Default: "AnyLayout".
   * @return Output of softmax. The data type and shape are the same as input .
   */
  Variable Softmax(const Variable& x, int axis = -1, const std::string& data_format = "AnyLayout");

  // *******************************************
  // Type converter Operator
  /**
   * @brief This OP takes in the Variable `x` with `x.dtype` and casts it to the output with dtype.
   * It’s meaningless if the output dtype equals the input `dtype`, but it’s fine if you do so.
   * @param x An input N-D variable.
   * @param dtype Data type of the output.
   * @return A variable with the same shape as input’s.
   */
  Variable Cast(const Variable& x, const std::string& dtype);

  // *******************************************
  // Decomposer Operator
  /**
   * @brief The gradient function of `elementwise_add`.
   * @param dout  The gradient variable of the `elementwise_add`'s output.
   * @param x The left input variable.
   * @param y The right input variable.
   * @param axis  Specify the axis to operate on the input. Default: -1.
   * @return The gradient variable of `x` and `y`.
   */
  const std::vector<Variable>& ElementwiseAddGrad(const Variable& dout,
                                                  const Variable& x,
                                                  const Variable& y,
                                                  int axis = -1);

  /**
   * @brief The gradient function of `relu`.
   * @param dout  The gradient variable of the `relu`'s output.
   * @param x The input variable.
   * @return The gradient variable of `x`.
   */
  Variable ReluGrad(const Variable& dout, const Variable& x);

  /**
   * @brief Compute the convolution.
   * @param x The image variable.
   * @param weight The filter variable.
   * @param strides The stride size. If stride is a list/tuple, it must contain two integers, (stride_H, stride_W).
   * Otherwise, the stride_H = stride_W = stride. Default: stride = {1, 1}.
   * @param paddings The padding size. If padding is a list/tuple, it must contain two integers, (padding_H, padding_W).
   * Otherwise, the padding_H = padding_W = padding. Default: padding = {0, 0}.
   * @param dilations  The dilation size. If dilation is a list/tuple, it must contain two integers, (dilation_H,
   * dilation_W). Otherwise, the dilation_H = dilation_W = dilation. Default: dilation = {1, 1}.
   * @param groups The groups number of the conv layer. Default: groups=1.
   * @param conv_type The convolution type. The choice contain "forward"/"backward_data"/"backward_filter", note only
   * support "forward" when using cudnn.
   * @param data_format Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
   * @param padding_algorithm CINN not support! It can be "EXPLICIT"/"SAME"/"VALID". Default: "EXPLICIT".
   * @param output_shape The shape of output. Default: None.
   * @return The convolution result variable.
   */
  Variable Conv(const Variable& x,
                const Variable& weight,
                const std::vector<int>& strides      = {1, 1},
                const std::vector<int>& paddings     = {0, 0},
                const std::vector<int>& dilations    = {1, 1},
                int groups                           = 1,
                const std::string& conv_type         = "forward",
                const std::string& data_format       = "NCHW",
                const std::string& padding_algorithm = "EXPLICIT",
                const std::vector<int>& output_shape = {});

  /**
   * @brief Compute the convolution-2d.
   * @param x The image variable.
   * @param weights The filter variable.
   * @param strides The stride size. If stride is a list/tuple, it must contain two integers, (stride_H, stride_W).
   * Otherwise, the stride_H = stride_W = stride. Default: stride = {1, 1}.
   * @param paddings The padding size. If padding is a list/tuple, it must contain two integers, (padding_H, padding_W).
   * Otherwise, the padding_H = padding_W = padding. Default: padding = {0, 0}.
   * @param dilations  The dilation size. If dilation is a list/tuple, it must contain two integers, (dilation_H,
   * dilation_W). Otherwise, the dilation_H = dilation_W = dilation. Default: dilation = {1, 1}.
   * @param groups The groups number of the conv layer. Default: groups=1.
   * @param data_format Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
   * @param padding_algorithm CINN not support! It can be "EXPLICIT"/"SAME"/"VALID". Default: "EXPLICIT".
   * @return The convolution-2d result variable.
   */
  Variable Conv2d(const Variable& x,
                  const Variable& weights,
                  const std::vector<int>& strides      = {1, 1},
                  const std::vector<int>& paddings     = {0, 0},
                  const std::vector<int>& dilations    = {1, 1},
                  int groups                           = 1,
                  const std::string& data_format       = "NCHW",
                  const std::string& padding_algorithm = "EXPLICIT");

  /**
   * @brief The gradient function of convolution-2d.
   * @param dout The gradient variable of the `conv2d`'s output.
   * @param x The image variable.
   * @param weights The filter variable.
   * @param strides The stride size. If stride is a list/tuple, it must contain two integers, (stride_H, stride_W).
   * Otherwise, the stride_H = stride_W = stride. Default: stride = {1, 1}.
   * @param paddings The padding size. If padding is a list/tuple, it must contain two integers, (padding_H, padding_W).
   * Otherwise, the padding_H = padding_W = padding. Default: padding = {0, 0}.
   * @param dilations  The dilation size. If dilation is a list/tuple, it must contain two integers, (dilation_H,
   * dilation_W). Otherwise, the dilation_H = dilation_W = dilation. Default: dilation = {1, 1}.
   * @param groups The groups number of the conv layer. Default: groups=1.
   * @param data_format Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
   * @param padding_algorithm CINN not support! It can be "EXPLICIT"/"SAME"/"VALID". Default: "EXPLICIT".
   * @return The gradient variable of 'x'.
   */
  std::vector<Variable> Conv2dGrad(const Variable& dout,
                                   const Variable& x,
                                   const Variable& weights,
                                   const std::vector<int>& strides      = {1, 1},
                                   const std::vector<int>& paddings     = {0, 0},
                                   const std::vector<int>& dilations    = {1, 1},
                                   const int groups                     = 1,
                                   const std::string& data_format       = "NCHW",
                                   const std::string& padding_algorithm = "EXPLICIT");

  /**
   * @brief Compute the depthwise convolution-2d.
   * @param x The image variable.
   * @param weights The filter variable.
   * @param strides The stride size. If stride is a list/tuple, it must contain two integers, (stride_H, stride_W).
   * Otherwise, the stride_H = stride_W = stride. Default: stride = {1, 1}.
   * @param paddings The padding size. If padding is a list/tuple, it must contain two integers, (padding_H, padding_W).
   * Otherwise, the padding_H = padding_W = padding. Default: padding = {0, 0}.
   * @param dilations  The dilation size. If dilation is a list/tuple, it must contain two integers, (dilation_H,
   * dilation_W). Otherwise, the dilation_H = dilation_W = dilation. Default: dilation = {1, 1}.
   * @param groups The groups number of the conv layer. Default: groups=1.
   * @param data_format Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
   * @param padding_algorithm CINN not support! It can be "EXPLICIT"/"SAME"/"VALID". Default: "EXPLICIT".
   * @return The depthwise convolution-2d result variable.
   */
  Variable DepthwiseConv2d(const Variable& x,
                           const Variable& weights,
                           const std::vector<int>& strides      = {1, 1},
                           const std::vector<int>& paddings     = {0, 0},
                           const std::vector<int>& dilations    = {1, 1},
                           int groups                           = 1,
                           const std::string& data_format       = "NCHW",
                           const std::string& padding_algorithm = "EXPLICIT");

  /**
   * @brief Compute the depthwise convolution-2d.
   * @param x The image variable.
   * @param scale Scale is a 1-dimensional tensor of size C that is applied to the output.
   * @param bias Bias is a 1-dimensional tensor of size C that is applied to the output.
   * @param mean The global mean (for training) or estimated mean (for testing).
   * @param variance The global variance (for training) or estimated Variance (for testing)
   * @param epsilon The small value added to the variance to prevent division by zero. Default: 1e-5f.
   * @param momentum The value used for the moving_mean and moving_var computation. Default: 0.9f.
   * @param data_layout Specify the input data format, may be “NC”, “NCL”, “NCHW”, “NCDHW”, “NLC”, “NHWC” or “NDHWC”.
   * Defalut “NCHW”.
   * @param is_test A flag indicating whether it is in test phrase or not.
   * @return `{out}` if `is_test` it true, `{out, saved_mean, saved_variance, moving_mean, moving_variance}` if
   * `is_test` is false.
   */
  std::vector<Variable> BatchNorm(const Variable& x,
                                  const Variable& scale,
                                  const Variable& bias,
                                  const Variable& mean,
                                  const Variable& variance,
                                  float epsilon                  = 1e-5f,
                                  float momentum                 = 0.9f,
                                  const std::string& data_layout = "NCHW",
                                  bool is_test                   = false);

  /**
   * @brief The gradient function of BatchNorm training.
   * @param dout The gradient variable of the `batch_norm_training`'s first output.
   * @param x The image variable.
   * @param scale Scale is a 1-dimensional tensor of size C that is applied to the output.
   * @param save_mean The global mean saved from forward compute.
   * @param save_variance The global variance from forward compute.
   * @param epsilon The small value added to the variance to prevent division by zero. Default: 1e-5f.
   * @param data_layout Specify the input data format, may be “NC”, “NCL”, “NCHW”, “NCDHW”, “NLC”, “NHWC” or “NDHWC”.
   * Defalut “NCHW”.
   * @return `{x_grad, scale_grad, bias_grad}`.
   */
  // batch norm grad, output(x_grad, scale_grad, bias_grad)
  std::vector<Variable> BatchNormGrad(const Variable& dout,
                                      const Variable& x,
                                      const Variable& scale,
                                      const Variable& save_mean,
                                      const Variable& save_variance,
                                      const float epsilon            = 1e-5f,
                                      const std::string& data_layout = "NCHW");

  /**
   * @brief Sort Variable x along the given axis. The original Variable x will not be changed.
   * @param operand The variable that will be sorted.
   * @param axis Specify the axis to operate on the input. Default: 0.
   * @param is_ascend Sort mode.
   * Defalut “NCHW”.
   * @return `Sorted variable index`.
   */
  Variable ArgSort(const Variable& operand, const int& axis, const bool& is_ascend = true);

  /**
   * @brief Sort Variable x along the given axis. The original Variable x will not be changed.
   * @param operand The variable that will be sorted.
   * @param axis Specify the axis to operate on the input. Default: 0.
   * @param is_ascend Sort mode.
   * Defalut “NCHW”.
   * @return `Sorted variable`.
   */
  Variable Sort(const Variable& operand, const int& axis, const bool& is_ascend = true);

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(NetBuilder);
};

}  // namespace frontend
}  // namespace cinn

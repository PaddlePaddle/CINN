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
#include <vector>

#include "cinn/common/macros.h"
#include "cinn/common/type.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/utils/type_defs.h"

namespace cinn {
namespace frontend {

#define NETBUILDER_UNARY_OP_FOREACH(macro__)                                                                 \
  macro__(Sqrt) macro__(Tanh) macro__(Relu) macro__(Sigmoid) macro__(Identity) macro__(Exp) macro__(Erf)     \
      macro__(Rsqrt) macro__(Log) macro__(Log2) macro__(Log10) macro__(Floor) macro__(Ceil) macro__(Round)   \
          macro__(Trunc) macro__(Sin) macro__(Cos) macro__(Tan) macro__(Sinh) macro__(Cosh) macro__(Asin)    \
              macro__(Acos) macro__(Atan) macro__(Asinh) macro__(Acosh) macro__(Atanh) macro__(IsNan)        \
                  macro__(IsFinite) macro__(IsInf) macro__(LogicalNot) macro__(BitwiseNot) macro__(Negative) \
                      macro__(Sign) macro__(Abs)

#define NETBUILDER_BINARY_OP_FOREACH(macro__)                                                               \
  macro__(Add) macro__(Sub) macro__(Div) macro__(Multiply) macro__(FloorDiv) macro__(Mod) macro__(FloorMod) \
      macro__(Max) macro__(Min) macro__(Power) macro__(LogicalAnd) macro__(LogicalOr) macro__(LogicalXor)   \
          macro__(BitwiseAnd) macro__(BitwiseOr) macro__(BitwiseXor) macro__(LeftShift) macro__(RightShift) \
              macro__(Equal) macro__(NotEqual) macro__(Greater) macro__(Less) macro__(GreaterEqual) macro__(LessEqual)

#define NETBUILDER_REDUCE_OP_FOREACH(macro__) \
  macro__(ReduceSum) macro__(ReduceProd) macro__(ReduceMax) macro__(ReduceMin) macro__(ReduceAll) macro__(ReduceAny)

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

  Placeholder CreateInput(const common::Type& type, const std::vector<int>& shape, const std::string& id_hint = "");
  Placeholder CreateInput(const Variable& input);

  // name of this builder
  const std::string& name() { return name_; }

  // the number of instructions
  const size_t size() { return instrs_.size(); }

  virtual ~NetBuilder() = default;

  void AppendInstruction(const Instruction& instr) { instrs_.push_back(instr); }

  void InferShape(Instruction instr) const;

 protected:
  /**
   * @brief The op only has one input and one output.
   *
   * @param operand The input variable.
   *
   * @return The result variable.
   */
  Variable UnaryOp(const std::string& op_type, const Variable& operand);

  Variable BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs, int axis = -1);

  /**
   * @brief Reduce array elements over the given dims.
   *
   * @param operand The input variable.
   * @param dim The dims along which a sum is performed. If dim is empty, the operation will sum over all elements
   * of the input array. If the dim has negative value, it should count from the last dim to the first.
   * @param keep_dim If it is set true, the axes which are reduced are left in the result as dimensions with size one.
   * With this option, the result will broadcast correctly against the input array.
   *
   * @return The result variable.
   */
  Variable Reduce(const std::string& op_type,
                  const Variable& operand,
                  const std::vector<int>& dim = {},
                  bool keep_dim               = false);

 private:
  // the helper function of Matmul
  std::pair<Variable, Variable> BroadcastMatmulInput(
      const Variable& x, const Variable& y, bool trans_x, bool trans_y, float alpha);
  std::vector<int> GetMatmulOutputShape(const Variable& x, const Variable& y, bool trans_x, bool trans_y, float alpha);

 public:
  const std::vector<Variable>& CustomInstr(const std::string& type,
                                           const std::vector<Variable>& inputs,
                                           const AttributeMap& attrs);

  // algorithm API
#define NETBUILDER_UNARY_OP_DECL(func_name__) Variable func_name__(const Variable& operand);
  NETBUILDER_UNARY_OP_FOREACH(NETBUILDER_UNARY_OP_DECL)
#undef NETBUILDER_UNARY_OP_DECL

#define NETBUILDER_ELEMENTWISE_OP_DECL(func_name__) \
  Variable func_name__(const Variable& lhs, const Variable& rhs, int axis = -1);
  NETBUILDER_ELEMENTWISE_OP_FOREACH(NETBUILDER_ELEMENTWISE_OP_DECL)
#undef NETBUILDER_ELEMENTWISE_OP_DECL

#define NETBUILDER_REDUCE_OP_DECL(func_name__) \
  Variable func_name__(const Variable& x, const std::vector<int>& dim = {}, bool keep_dim = false);
  NETBUILDER_REDUCE_OP_FOREACH(NETBUILDER_REDUCE_OP_DECL)
#undef NETBUILDER_REDUCE_OP_DECL

  Variable ConstScalar(float value, const std::string& name, const std::string& dtype);

  /**
   * @brief Create scalar with the specific value and type.
   * @param value The scalar value to be set.
   * @param name The name of output variable.
   * @return The result variable.
   */
  template <typename T>
  Variable ConstScalar(T value, const std::string& name) {
    return ConstScalar(static_cast<float>(value), name, common::Type2Str(common::type_of<T>()));
  }

  Variable FillConstant(const std::vector<int>& shape,
                        float value,
                        const std::string& name,
                        const std::string& dtype,
                        bool force_cpu = false);

  template <typename T = float>
  Variable FillConstant(const std::vector<int>& shape, T value, const std::string& name, bool force_cpu = false) {
    return FillConstant(shape, static_cast<float>(value), name, common::Type2Str(common::type_of<T>()), force_cpu);
  }

  Variable BroadcastTo(const Variable& operand, const std::vector<int>& out_shape);

  Variable BroadcastTo(const Variable& operand,
                       const std::vector<int>& out_shape,
                       const std::vector<int>& broadcast_axes);

  Variable Concat(const std::vector<Variable>& input_vars, int axis = 0);

  std::vector<Variable> Split(const Variable& operand, const std::vector<int>& num_or_sections, int axis = 0);

  Variable Reshape(const Variable& operand, const std::vector<int>& shape);

  /**
   * This API reverses the Variable x along the given axis.
   * Example 1: x = [[0, 1], [2, 3], [4, 5]], axis = [0]
   *            output = [[4, 5], [2, 3], [0, 1]]
   * Example 2: x = [[0, 1], [2, 3], [4, 5]], axis = [0, 1]
   *            output = [[5, 4], [3, 2], [1, 0]]
   */
  Variable Reverse(const Variable& operand, const std::vector<int>& axis);

  Variable Transpose(const Variable& operand, const std::vector<int>& axis);

  Variable Slice(const Variable& operand,
                 const std::vector<int>& axes,
                 const std::vector<int>& starts      = {},
                 const std::vector<int>& ends        = {},
                 const std::vector<int>& infer_flags = {},
                 const std::vector<int>& strides     = {});

  Variable Select(const Variable& condition, const Variable& true_value, const Variable& false_value);

  Variable IndexSelect(const Variable& operand, const Variable& index, int axis = 0);

  Variable ScatterAssign(const Variable& operand, const Variable& updates, const Variable& index, int axis = 0);

  Variable ScatterAdd(const Variable& operand, const Variable& updates, const Variable& index, int axis = 0);

  Variable SliceAssign(const Variable& input,
                       const Variable& assign,
                       const std::vector<int>& axes,
                       const std::vector<int>& starts,
                       const std::vector<int>& ends,
                       const std::vector<int>& strides = {});

  /**
   * Multiply two matrix.
   */
  Variable Mul(const Variable& a, const Variable& b, int x_num_col_dims = 1, int y_num_col_dims = 1);

  Variable Scale(const Variable& a, float scale = 1.0f, float bias = 0.0f, bool bias_after_scale = true);

  Variable Sum(const std::vector<Variable>& inputs);

  // Matmul not MatMul is not mistake, the SnakeName function in pybind need this name type
  Variable Matmul(const Variable& x, const Variable& y, bool trans_x = false, bool trans_y = false, float alpha = 1.0f);

  /**
   * The gradient of elementwise_add.
   */
  const std::vector<Variable>& ElementwiseAddGrad(const Variable& dout,
                                                  const Variable& x,
                                                  const Variable& y,
                                                  int axis = -1);

  Variable Relu6(const Variable& a, float threshold = 6.0f);

  Variable ReluGrad(const Variable& lhs, const Variable& rhs);

  /**
   * Cast Variable x to dtype.
   */
  Variable Cast(const Variable& operand, const std::string& dtype);

  /**
   * Squeeze Variable x along the given axes.
   */
  Variable Squeeze(const Variable& operand, const std::vector<int>& axes);

  Variable Conv(const Variable& lhs,
                const Variable& rhs,
                const std::vector<int>& strides      = {1, 1},
                const std::vector<int>& paddings     = {0, 0},
                const std::vector<int>& dilations    = {1, 1},
                int groups                           = 1,
                const std::string& conv_type         = "forward",
                const std::string& data_format       = "NCHW",
                const std::string& padding_algorithm = "EXPLICIT",
                const std::vector<int>& output_shape = {});

  /**
   * The convolution2D layer calculates the output based on the input, filter
   * and strides, paddings, dilations, groups parameters.
   */
  Variable Conv2d(const Variable& a,
                  const Variable& b,
                  const std::vector<int>& strides      = {1, 1},
                  const std::vector<int>& paddings     = {0, 0},
                  const std::vector<int>& dilations    = {1, 1},
                  int groups                           = 1,
                  const std::string& data_format       = "NCHW",
                  const std::string& padding_algorithm = "EXPLICIT");

  // conv2d grad, output(grad_x, grad_w)
  std::vector<Variable> Conv2dGrad(const Variable& dy,
                                   const Variable& x,
                                   const Variable& w,
                                   const std::vector<int>& strides      = {1, 1},
                                   const std::vector<int>& paddings     = {0, 0},
                                   const std::vector<int>& dilations    = {1, 1},
                                   const int groups                     = 1,
                                   const std::string& data_format       = "NCHW",
                                   const std::string& padding_algorithm = "EXPLICIT");

  Variable DepthwiseConv2d(const Variable& a,
                           const Variable& b,
                           const std::vector<int>& strides      = {1, 1},
                           const std::vector<int>& paddings     = {0, 0},
                           const std::vector<int>& dilations    = {1, 1},
                           int groups                           = 1,
                           const std::string& data_format       = "NCHW",
                           const std::string& padding_algorithm = "EXPLICIT");

  Variable Pool2d(const Variable& a,
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
   * is_test(true): batch norm infer (default), output={y}
   * is_test(false): batch norm training, outputs={y, saved_mean, saved_variance, moving_mean, moving_variance}
   */
  std::vector<Variable> BatchNorm(const Variable& a,
                                  const Variable& scale,
                                  const Variable& bias,
                                  const Variable& mean,
                                  const Variable& variance,
                                  float epsilon                  = 1e-5f,
                                  float momentum                 = 0.9f,
                                  const std::string& data_layout = "NCHW",
                                  bool is_test                   = false);

  // batch norm grad, output(x_grad, scale_grad, bias_grad)
  std::vector<Variable> BatchNormGrad(const Variable& dy,
                                      const Variable& x,
                                      const Variable& scale,
                                      const Variable& save_mean,
                                      const Variable& save_variance,
                                      const float epsilon            = 1e-5,
                                      const std::string& data_layout = "NCHW");

  Variable Softmax(const Variable& a, int axis = -1, const std::string& data_format = "AnyLayout");

  Variable DropoutInfer(const Variable& a,
                        float dropout_prob                        = 0.5f,
                        const std::string& dropout_implementation = "downgrade_in_infer");

  Variable Clip(const std::vector<Variable>& inputs, const float& max_val, const float& min_val);

  Variable Arange(const float start, const float stop, const float step, const std::string& dtype);

  // This operator checks if all x and y satisfy the condition: |x - y| <= atol + rtol * |y|
  Variable IsClose(
      const Variable& x, const Variable& y, float rtol = 1e-05f, float atol = 1e-08f, bool equal_nan = false);

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(NetBuilder);
};

}  // namespace frontend
}  // namespace cinn

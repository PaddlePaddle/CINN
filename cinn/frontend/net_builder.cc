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

#include "cinn/frontend/net_builder.h"

#include <string>
#include <utility>
#include <vector>

#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

using common::Context;
using common::Type;
using hlir::framework::Operator;
using utils::AttributeMap;
using utils::ShapeType;

NetBuilder::NetBuilder(const std::string& name) : name_(name) {}

Program NetBuilder::Build(bool in_reverse) {
  std::vector<Instruction> instrs;
  if (in_reverse) {
    instrs.reserve(instrs_.size());
    for (auto it = instrs_.rbegin(); it != instrs_.rend(); it++) {
      instrs.emplace_back(*it);
    }
  } else {
    instrs = std::move(instrs_);
  }

  Program program{std::move(instrs), std::move(inputs_)};
  program.Validate();
  return program;
}

void NetBuilder::InferShape(Instruction instr) const {
  using ShapeFunc           = std::function<std::vector<ShapeType>(const std::vector<ShapeType>&, const AttributeMap&)>;
  using TypeFunc            = std::function<std::vector<Type>(const std::vector<Type>&, const AttributeMap&)>;
  const auto& op_infershape = Operator::GetAttrs<ShapeFunc>("infershape");
  const auto& op_inferdtype = Operator::GetAttrs<TypeFunc>("inferdtype");

  size_t size = instr->inputs.size();
  std::vector<ShapeType> in_shapes(size);
  std::vector<Type> in_types(size);
  std::transform(
      instr->inputs.begin(), instr->inputs.end(), in_shapes.begin(), [](const Variable& var) { return var->shape; });
  std::transform(
      instr->inputs.begin(), instr->inputs.end(), in_types.begin(), [](const Variable& var) { return var->type; });

  auto key        = Operator::Get(instr->op_type);
  auto out_shapes = op_infershape[key](in_shapes, instr->attrs);
  auto out_types  = op_inferdtype[key](in_types, instr->attrs);

  auto& outs            = instr->outputs;
  size_t origin_out_num = outs.size();
  outs.resize(out_shapes.size());
  for (size_t i = origin_out_num; i < outs.size(); i++) {
    outs[i] = Variable();
  }
  for (size_t i = 0; i < outs.size(); i++) {
    outs[i]->shape = out_shapes[i];
    outs[i]->type  = out_types[i];
  }
}

const std::vector<Variable>& NetBuilder::CustomInstr(const std::string& type,
                                                     const std::vector<Variable>& inputs,
                                                     const AttributeMap& attrs) {
  Instruction instr(type, inputs);
  for (auto& kv : attrs) {
    instr.SetAttr(kv.first, kv.second);
  }

  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable NetBuilder::BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs, int axis) {
  return CustomInstr(op_type, {lhs, rhs}, {{"axis", axis}}).front();
}

Variable NetBuilder::UnaryOp(const std::string& op_type, const Variable& operand) {
  return CustomInstr(op_type, {operand}, {}).front();
}

Variable NetBuilder::Reduce(const std::string& op_type, const Variable& x, const std::vector<int>& dim, bool keep_dim) {
  return CustomInstr(op_type, {x}, {{"dim", dim}, {"keep_dim", keep_dim}}).front();
}

#define NETBUILDER_UNARY_OP_DEF(func_name__, op_type__) \
  Variable NetBuilder::func_name__(const Variable& operand) { return UnaryOp(#op_type__, operand); }
NETBUILDER_UNARY_OP_DEF(Sqrt, sqrt)
NETBUILDER_UNARY_OP_DEF(Tanh, tanh)
NETBUILDER_UNARY_OP_DEF(Relu, relu)
NETBUILDER_UNARY_OP_DEF(Gelu, gelu)
NETBUILDER_UNARY_OP_DEF(Sigmoid, sigmoid)
NETBUILDER_UNARY_OP_DEF(Identity, identity)
NETBUILDER_UNARY_OP_DEF(Exp, exp)
NETBUILDER_UNARY_OP_DEF(Erf, erf)
NETBUILDER_UNARY_OP_DEF(Rsqrt, rsqrt)
NETBUILDER_UNARY_OP_DEF(Log, log)
NETBUILDER_UNARY_OP_DEF(Log2, log2)
NETBUILDER_UNARY_OP_DEF(Log10, log10)
NETBUILDER_UNARY_OP_DEF(Floor, floor)
NETBUILDER_UNARY_OP_DEF(Ceil, ceil)
NETBUILDER_UNARY_OP_DEF(Round, round)
NETBUILDER_UNARY_OP_DEF(Trunc, trunc)
NETBUILDER_UNARY_OP_DEF(Sin, sin)
NETBUILDER_UNARY_OP_DEF(Cos, cos)
NETBUILDER_UNARY_OP_DEF(Tan, tan)
NETBUILDER_UNARY_OP_DEF(Sinh, sinh)
NETBUILDER_UNARY_OP_DEF(Cosh, cosh)
NETBUILDER_UNARY_OP_DEF(Asin, asin)
NETBUILDER_UNARY_OP_DEF(Acos, acos)
NETBUILDER_UNARY_OP_DEF(Atan, atan)
NETBUILDER_UNARY_OP_DEF(Asinh, asinh)
NETBUILDER_UNARY_OP_DEF(Acosh, acosh)
NETBUILDER_UNARY_OP_DEF(Atanh, atanh)
NETBUILDER_UNARY_OP_DEF(IsNan, isnan)
NETBUILDER_UNARY_OP_DEF(IsFinite, isfinite)
NETBUILDER_UNARY_OP_DEF(IsInf, isinf)
NETBUILDER_UNARY_OP_DEF(LogicalNot, logical_not)
NETBUILDER_UNARY_OP_DEF(BitwiseNot, bitwise_not)
NETBUILDER_UNARY_OP_DEF(Negative, negative)
NETBUILDER_UNARY_OP_DEF(Sign, sign)
NETBUILDER_UNARY_OP_DEF(Abs, abs)

#undef NETBUILDER_UNARY_OP_DEF

#define NETBUILDER_BINARY_OP_DEF(func_name__, op_type__)                                 \
  Variable NetBuilder::func_name__(const Variable& lhs, const Variable& rhs, int axis) { \
    return BinaryOp(#op_type__, lhs, rhs, axis);                                         \
  }
NETBUILDER_BINARY_OP_DEF(Add, elementwise_add)
NETBUILDER_BINARY_OP_DEF(Multiply, elementwise_mul)
NETBUILDER_BINARY_OP_DEF(Divide, divide)
NETBUILDER_BINARY_OP_DEF(Subtract, substract)
NETBUILDER_BINARY_OP_DEF(FloorDivide, floor_divide)
NETBUILDER_BINARY_OP_DEF(Mod, mod)
NETBUILDER_BINARY_OP_DEF(FloorMod, floor_mod)
NETBUILDER_BINARY_OP_DEF(Max, max)
NETBUILDER_BINARY_OP_DEF(Min, min)
NETBUILDER_BINARY_OP_DEF(Pow, pow)
NETBUILDER_BINARY_OP_DEF(LogicalAnd, logical_and)
NETBUILDER_BINARY_OP_DEF(LogicalOr, logical_or)
NETBUILDER_BINARY_OP_DEF(LogicalXor, logical_xor)
NETBUILDER_BINARY_OP_DEF(BitwiseAnd, bitwise_and)
NETBUILDER_BINARY_OP_DEF(BitwiseOr, bitwise_or)
NETBUILDER_BINARY_OP_DEF(BitwiseXor, bitwise_xor)
NETBUILDER_BINARY_OP_DEF(LeftShift, left_shift)
NETBUILDER_BINARY_OP_DEF(RightShift, right_shift)
NETBUILDER_BINARY_OP_DEF(GreaterThan, greater);
NETBUILDER_BINARY_OP_DEF(LessThan, less);
NETBUILDER_BINARY_OP_DEF(Equal, equal);
NETBUILDER_BINARY_OP_DEF(NotEqual, not_equal);
NETBUILDER_BINARY_OP_DEF(GreaterEqual, greater_equal);
NETBUILDER_BINARY_OP_DEF(LessEqual, less_equal);

#undef NETBUILDER_BINARY_OP_DEF

#define NETBUILDER_REDUCE_OP_DEF(func_name__, op_type__)                                            \
  Variable NetBuilder::func_name__(const Variable& x, const std::vector<int>& dim, bool keep_dim) { \
    return Reduce(#op_type__, x, dim, keep_dim);                                                    \
  }

NETBUILDER_REDUCE_OP_DEF(ReduceSum, reduce_sum)
NETBUILDER_REDUCE_OP_DEF(ReduceProd, reduce_prod)
NETBUILDER_REDUCE_OP_DEF(ReduceMax, reduce_max)
NETBUILDER_REDUCE_OP_DEF(ReduceMin, reduce_min)
NETBUILDER_REDUCE_OP_DEF(ReduceAll, reduce_all)
NETBUILDER_REDUCE_OP_DEF(ReduceAny, reduce_any)

#undef NETBUILDER_REDUCE_OP_DEF

Placeholder NetBuilder::CreateInput(const Type& type, const std::vector<int>& shape, const std::string& id_hint) {
  if (!id_hint.empty()) {
    CheckVarNameValid(id_hint);
  }
  std::string id = id_hint.empty() ? Context::Global().NewName("placeholder") : id_hint;

  inputs_.emplace_back(id);
  auto& var  = inputs_.back();
  var->type  = type;
  var->shape = shape;
  return Placeholder(var);
}

Placeholder NetBuilder::CreateInput(const Variable& var) {
  CHECK(!var->shape.empty()) << "The input's shape is not set yet";
  CHECK(!var->type.is_unk()) << "The input's type is not set yet";
  inputs_.push_back(var);
  return Placeholder(var);
}

Variable NetBuilder::FillConstant(
    const std::vector<int>& shape, float value, const std::string& name, const std::string& dtype, bool force_cpu) {
  auto out =
      CustomInstr("fill_constant", {}, {{"shape", shape}, {"value", value}, {"dtype", dtype}, {"force_cpu", force_cpu}})
          .front();
  out.set_id(name);
  return out;
}

std::vector<Variable> NetBuilder::Split(const Variable& operand, const std::vector<int>& num_or_sections, int axis) {
  return CustomInstr("split", {operand}, {{"num_or_sections", num_or_sections}, {"axis", axis}});
}

Variable NetBuilder::Concat(const std::vector<Variable>& input_vars, int axis) {
  CHECK(!input_vars.empty()) << "The inputs of concat op should not be empty! Please check.";
  if (input_vars.size() == 1UL) {
    return Identity(input_vars.front());
  }
  return CustomInstr("concat", input_vars, {{"axis", axis}}).front();
}

Variable NetBuilder::BroadcastTo(const Variable& operand, const std::vector<int>& out_shape) {
  auto x_shape_size = operand->shape.size();
  auto y_shape_size = out_shape.size();
  CHECK_GT(x_shape_size, 0) << "Cannot broadcast a empty operand " << operand->id << " to "
                            << cinn::utils::Join(out_shape, ",");
  CHECK_LE(x_shape_size, y_shape_size) << "The broadcast_p's input shape dimension should less than the output's, "
                                       << "but here (" << x_shape_size << " > " << y_shape_size << ").";

  VLOG(4) << "Try broadcast " << operand->id << " from shape (" << cinn::utils::Join(operand->shape, ",")
          << ") to shape (" << cinn::utils::Join(out_shape, ",") << ")";

  std::vector<int> broadcast_axes(x_shape_size, 0);
  if (x_shape_size > 1) {
    for (int i = 1; i <= x_shape_size; ++i) {
      CHECK((out_shape[y_shape_size - i] == operand->shape[x_shape_size - i]) ||
            (operand->shape[x_shape_size - i] == 1))
          << "We cannot broadcast from shape (" << cinn::utils::Join(operand->shape, ",") << ") to shape ("
          << cinn::utils::Join(out_shape, ",") << ")";
      broadcast_axes[x_shape_size - i] = y_shape_size - i;
    }
  } else {
    int axis     = -1;
    auto x_shape = operand->shape.at(0);
    if (x_shape == 1) {
      // Can broadcast directly, default axis 0
      axis = 0;
    } else {
      // The broadcast axes is the index of the shape in out_shape when the input dimension is 1
      for (int i = 0; i < y_shape_size; ++i) {
        if (out_shape[i] == x_shape) {
          axis = i;
          break;
        }
      }
      CHECK_NE(axis, -1) << "When we broadcast a 1-dimension shape, the number should contained in the out_shape. "
                         << "We cannot broadcast from shape (" << cinn::utils::Join(operand->shape, ",")
                         << ") to shape (" << cinn::utils::Join(out_shape, ",") << ")";
    }
    broadcast_axes[0] = axis;
  }

  return BroadcastTo(operand, out_shape, broadcast_axes);
}

Variable NetBuilder::BroadcastTo(const Variable& operand,
                                 const std::vector<int>& out_shape,
                                 const std::vector<int>& broadcast_axes) {
  return CustomInstr("broadcast_to", {operand}, {{"out_shape", out_shape}, {"broadcast_axes", broadcast_axes}}).front();
}

Variable NetBuilder::Reshape(const Variable& operand, const std::vector<int>& shape) {
  return CustomInstr("reshape", {operand}, {{"shape", shape}}).front();
}

Variable NetBuilder::Transpose(const Variable& operand, const std::vector<int>& axis) {
  return CustomInstr("transpose", {operand}, {{"axis", axis}}).front();
}

Variable NetBuilder::Slice(const Variable& operand,
                           const std::vector<int>& axes,
                           const std::vector<int>& starts,
                           const std::vector<int>& ends,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& strides) {
  return CustomInstr(
             "slice",
             {operand},
             {{"axes", axes}, {"starts", starts}, {"ends", ends}, {"infer_flags", infer_flags}, {"strides", strides}})
      .front();
}

Variable NetBuilder::SliceAssign(const Variable& input,
                                 const Variable& assign,
                                 const std::vector<int>& axes,
                                 const std::vector<int>& starts,
                                 const std::vector<int>& ends,
                                 const std::vector<int>& strides) {
  return CustomInstr("slice_assign",
                     {input, assign},
                     {{"axes", axes}, {"starts", starts}, {"ends", ends}, {"strides", strides}})
      .front();
}

Variable NetBuilder::Reverse(const Variable& operand, const std::vector<int>& axis) {
  return CustomInstr("reverse", {operand}, {{"axis", axis}}).front();
}

Variable NetBuilder::Select(const Variable& condition, const Variable& true_value, const Variable& false_value) {
  return CustomInstr("select", {condition, true_value, false_value}, {}).front();
}

Variable NetBuilder::IndexSelect(const Variable& operand, const Variable& index, int axis) {
  return CustomInstr("index_select", {operand, index}, {{"axis", axis}}).front();
}

Variable NetBuilder::ScatterAssign(const Variable& operand, const Variable& updates, const Variable& index, int axis) {
  return CustomInstr("scatter_assign", {operand, updates, index}, {{"axis", axis}}).front();
}

Variable NetBuilder::ScatterAdd(const Variable& operand, const Variable& updates, const Variable& index, int axis) {
  return CustomInstr("scatter_add", {operand, updates, index}, {{"axis", axis}}).front();
}

Variable NetBuilder::IsClose(const Variable& x, const Variable& y, float rtol, float atol, bool equal_nan) {
  return CustomInstr("isclose", {x, y}, {{"rtol", rtol}, {"atol", atol}, {"equal_nan", equal_nan}}).front();
}

Variable NetBuilder::Mul(const Variable& a, const Variable& b, int x_num_col_dims, int y_num_col_dims) {
  return CustomInstr("mul", {a, b}, {{"x_num_col_dims", x_num_col_dims}, {"y_num_col_dims", y_num_col_dims}}).front();
}

const std::vector<Variable>& NetBuilder::ElementwiseAddGrad(const Variable& dout,
                                                            const Variable& x,
                                                            const Variable& y,
                                                            int axis) {
  return CustomInstr("elementwise_add_grad", {dout, x, y}, {{"axis", axis}});
}

Variable NetBuilder::Relu6(const Variable& a, float threshold) {
  return CustomInstr("relu6", {a}, {{"threshold", threshold}}).front();
}

Variable NetBuilder::ReluGrad(const Variable& lhs, const Variable& rhs) {
  return CustomInstr("relu_grad", {lhs, rhs}, {}).front();
}

Variable NetBuilder::Gather(const Variable& x, const Variable& index, const int& axis) {
  return CustomInstr("gather", {x, index}, {{"axis", axis}}).front();
}

Variable NetBuilder::GatherNd(const Variable& x, const Variable& index, const std::vector<int>& axes) {
  return CustomInstr("gather_nd", {x, index}, {{"axes", axes}}).front();
}

Variable NetBuilder::Scatter(const Variable& src, const Variable& index, const Variable& out, const int& axis) {
  return CustomInstr("scatter", {src, index, out}, {{"axis", axis}}).front();
}
Variable NetBuilder::Scatter(const Variable& src,
                             const Variable& index,
                             const std::vector<int>& shape,
                             const float& default_value,
                             const int& axis) {
  auto out = FillConstant(shape, default_value, UniqName("fill_constant"), "float", false);
  return Scatter(src, index, out, axis);
}

Variable NetBuilder::ScatterNd(const Variable& src,
                               const Variable& index,
                               const Variable& out,
                               const std::vector<int>& axes) {
  return CustomInstr("scatter_nd", {src, index, out}, {{"axes", axes}}).front();
}
Variable NetBuilder::ScatterNd(const Variable& src,
                               const Variable& index,
                               const std::vector<int>& shape,
                               const float& default_value,
                               const std::vector<int>& axes) {
  auto out = FillConstant(shape, default_value, UniqName("fill_constant"), "float", false);
  return ScatterNd(src, index, out, axes);
}

Variable NetBuilder::Cast(const Variable& operand, const std::string& dtype) {
  if (operand->type == common::Str2Type(dtype)) {
    return Identity(operand);
  }
  return CustomInstr("cast", {operand}, {{"dtype", dtype}}).front();
}

Variable NetBuilder::OneHot(const Variable& indices,
                            const Variable& on_value,
                            const Variable& off_value,
                            const int depth,
                            const int axis,
                            const std::string& dtype) {
  return CustomInstr("one_hot", {indices, on_value, off_value}, {{"depth", depth}, {"axis", axis}, {"dtype", dtype}})
      .front();
}

Variable NetBuilder::Squeeze(const Variable& operand, const std::vector<int>& axes) {
  return CustomInstr("squeeze", {operand}, {{"axes", axes}}).front();
}

Variable NetBuilder::ExpandDims(const Variable& operand, int axis, int num_newaxis) {
  return CustomInstr("expand_dims", {operand}, {{"axis", axis}, {"num_newaxis", num_newaxis}}).front();
}

Variable NetBuilder::Conv(const Variable& lhs,
                          const Variable& rhs,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations,
                          int groups,
                          const std::string& conv_type,
                          const std::string& data_format,
                          const std::string& padding_algorithm,
                          const std::vector<int>& output_shape) {
  return CustomInstr("conv2d",
                     {lhs, rhs},
                     {{"stride", strides},
                      {"padding", paddings},
                      {"dilation", dilations},
                      {"groups", groups},
                      {"conv_type", conv_type},
                      {"data_format", data_format},
                      {"padding_algorithm", padding_algorithm},
                      {"output_shape", output_shape}})
      .front();
}

Variable NetBuilder::ArgSort(const Variable& operand, const int& axis, const bool& is_ascend) {
  Instruction instr("argsort", {operand});
  instr.SetAttr("axis", axis);
  instr.SetAttr("is_ascend", is_ascend);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Sort(const Variable& operand, const int& axis, const bool& is_ascend) {
  Instruction instr("sort", {operand});
  instr.SetAttr("axis", axis);
  instr.SetAttr("is_ascend", is_ascend);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Argmax(const Variable& x, const int& axis, const bool& keep_dim) {
  Instruction instr("argmax", {x});
  instr.SetAttr("axis", axis);
  instr.SetAttr("keep_dim", keep_dim);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Argmin(const Variable& x, const int& axis, const bool& keep_dim) {
  Instruction instr("argmin", {x});
  instr.SetAttr("axis", axis);
  instr.SetAttr("keep_dim", keep_dim);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Conv2d(const Variable& a,
                            const Variable& b,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& dilations,
                            int groups,
                            const std::string& data_format,
                            const std::string& padding_algorithm) {
  return Conv(a, b, strides, paddings, dilations, groups, "forward", data_format, padding_algorithm, {});
}

Variable NetBuilder::DepthwiseConv2d(const Variable& a,
                                     const Variable& b,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings,
                                     const std::vector<int>& dilations,
                                     int groups,
                                     const std::string& data_format,
                                     const std::string& padding_algorithm) {
  return CustomInstr("depthwise_conv2d",
                     {a, b},
                     {{"stride", strides},
                      {"padding", paddings},
                      {"dilation", dilations},
                      {"groups", groups},
                      {"data_format", data_format},
                      {"padding_algorithm", padding_algorithm}})
      .front();
}

Variable NetBuilder::Pool2d(const Variable& a,
                            const std::string& pooling_type,
                            const std::vector<int>& ksize,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            bool ceil_mode,
                            bool exclusive,
                            bool global_pooling,
                            const std::string& data_format,
                            bool adaptive,
                            const std::string& padding_algorithm) {
  return CustomInstr("pool2d",
                     {a},
                     {{"pool_type", pooling_type},
                      {"kernel_size", ksize},
                      {"stride_size", strides},
                      {"padding_size", paddings},
                      {"ceil_mode", ceil_mode},
                      {"exclusive", exclusive},
                      {"global_pooling", global_pooling},
                      {"data_format", data_format},
                      {"adaptive", adaptive},
                      {"padding_algorithm", padding_algorithm}})
      .front();
}

Variable NetBuilder::Repeat(const Variable& x, int repeats, int axis) {
  return CustomInstr("repeat", {x}, {{"repeats", repeats}, {"axis", axis}}).front();
}

std::vector<Variable> NetBuilder::BatchNorm(const Variable& a,
                                            const Variable& scale,
                                            const Variable& bias,
                                            const Variable& mean,
                                            const Variable& variance,
                                            float epsilon,
                                            float momentum,
                                            const std::string& data_layout,
                                            bool is_test) {
  std::string op_type = is_test ? "batch_norm" : "batch_norm_train";
  return CustomInstr(op_type,
                     {a, scale, bias, mean, variance},
                     {{"epsilon", epsilon}, {"momentum", momentum}, {"data_layout", data_layout}});
}

// batch norm grad, output(grad_x, grad_scale, grad_bias)
std::vector<Variable> NetBuilder::BatchNormGrad(const Variable& dy,
                                                const Variable& x,
                                                const Variable& scale,
                                                const Variable& save_mean,
                                                const Variable& save_variance,
                                                const float epsilon,
                                                const std::string& data_layout) {
  return CustomInstr("batch_norm_grad",
                     {dy, x, scale, save_mean, save_variance},
                     {{"epsilon", epsilon}, {"data_layout", data_layout}});
}

Variable NetBuilder::Scale(const Variable& a, float scale, float bias, bool bias_after_scale) {
  return CustomInstr("scale", {a}, {{"scale", scale}, {"bias", bias}, {"bias_after_scale", bias_after_scale}}).front();
}

Variable NetBuilder::Softmax(const Variable& a, int axis, const std::string& data_format) {
  return CustomInstr("softmax", {a}, {{"axis", axis}, {"data_format", data_format}}).front();
}

Variable NetBuilder::DropoutInfer(const Variable& a, float dropout_prob, const std::string& dropout_implementation) {
  return CustomInstr(
             "dropout_infer", {a}, {{"dropout_prob", dropout_prob}, {"dropout_implementation", dropout_implementation}})
      .front();
}

Variable NetBuilder::Sum(const std::vector<Variable>& inputs) {
  return CustomInstr("sum", inputs, {}).front();
  ;
}

Variable NetBuilder::Clip(const std::vector<Variable>& inputs, const float& max_val, const float& min_val) {
  return CustomInstr("clip", inputs, {{"max_val", max_val}, {"min_val", min_val}}).front();
}

Variable NetBuilder::Arange(const float start, const float stop, const float step, const std::string& dtype) {
  return CustomInstr("arange", {}, {{"start", start}, {"stop", stop}, {"step", step}, {"dtype", dtype}}).front();
}

Variable NetBuilder::Flip(const Variable& operand, const std::vector<int>& axes) {
  Instruction instr("flip", {operand});
  instr.SetAttr("axes", axes);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

// conv2d grad, output(grad_x, grad_w)
std::vector<Variable> NetBuilder::Conv2dGrad(const Variable& dy,
                                             const Variable& x,
                                             const Variable& w,
                                             const std::vector<int>& strides,
                                             const std::vector<int>& paddings,
                                             const std::vector<int>& dilations,
                                             const int groups,
                                             const std::string& data_format,
                                             const std::string& padding_algorithm) {
  return CustomInstr("conv2d_grad",
                     {dy, x, w},
                     {{"strides", strides},
                      {"paddings", paddings},
                      {"dilations", dilations},
                      {"groups", groups},
                      {"data_format", data_format},
                      {"padding_algorithm", padding_algorithm}});
}

std::pair<Variable, Variable> NetBuilder::BroadcastMatmulInput(
    const Variable& x, const Variable& y, bool trans_x, bool trans_y, float alpha) {
  const auto &x_shape = x->shape, &y_shape = y->shape;

  auto matmul_info = [&]() {
    std::stringstream ss;
    ss << "matmul(X:" << x->id << "[" << cinn::utils::Join(x_shape, ", ") << "], Y:" << y->id << "["
       << cinn::utils::Join(y_shape, ", ") << "]"
       << ", trans_x=" << trans_x << ", trans_y=" << trans_y << ", alpha=" << alpha << ")";
    return ss.str();
  };

  CHECK(!x_shape.empty()) << "The input X:" << x->id << " of matmul should not empty! Please check.";
  CHECK(!y_shape.empty()) << "The input Y:" << y->id << " of matmul should not empty! Please check.";

  int x_dim = x_shape.size(), y_dim = y_shape.size();
  int max_dim = std::max(x_shape.size(), y_shape.size());

  std::vector<int> new_x_shape, new_y_shape;
  if (max_dim == 1) {
    // vector * vector
    CHECK(x_shape == y_shape)
        << "The matmul input X's numbers must be equal to Y's numbers,when X/Y's dims =1. But here " << matmul_info();

    // do not need broadcast
    return {x, y};
  } else if (x_dim == 1) {
    // vector * matrix
    int y_K = trans_y ? y_shape[max_dim - 1] : y_shape[max_dim - 2];
    CHECK_EQ(y_K, x_shape[0]) << "The K dimension of Y:" << y_K << " should equal to X.shape[0]:" << x_shape[0]
                              << ". But here " << matmul_info();

    // broadcast vector x to the same batch size
    // [m] * [a, b, m, d] -> [a, b, 1, m] * [a, b, m, d]
    new_x_shape              = y_shape;
    new_x_shape[max_dim - 2] = 1;
    new_x_shape[max_dim - 1] = x_shape[0];
  } else if (y_dim == 1) {
    // matrix * vector
    int x_K = trans_x ? x_shape[max_dim - 2] : x_shape[max_dim - 1];
    CHECK_EQ(x_K, y_shape[0]) << "The K dimension of X:" << x_K << " should equal to Y.shape[0]:" << y_shape[0]
                              << ". But here " << matmul_info();

    // broadcast vector y to the same batch size
    // [a, b, c, m] * [m] -> [a, b, c, m] * [a, b, m, 1]
    new_y_shape              = x_shape;
    new_y_shape[max_dim - 2] = y_shape[0];
    new_y_shape[max_dim - 1] = 1;
  } else {
    // matrix * matrix
    int x_K = trans_x ? x_shape[x_dim - 2] : x_shape[x_dim - 1];
    int y_K = trans_y ? y_shape[y_dim - 1] : y_shape[y_dim - 2];
    CHECK_EQ(x_K, y_K) << "The K dimension of matmul not equal. Where " << matmul_info();

    // if dimension of A or B greater than 2, broadcast input to the same shape
    auto gen_new_shape = [max_dim](const std::vector<int>& old_shape) {
      std::vector<int> new_shape;
      if (old_shape.size() != max_dim) {
        // if dim not equal, full 1
        new_shape.resize(max_dim - old_shape.size(), 1);
        new_shape.insert(new_shape.end(), old_shape.begin(), old_shape.end());
      } else {
        new_shape = old_shape;
      }
      return new_shape;
    };
    new_x_shape = gen_new_shape(x_shape);
    new_y_shape = gen_new_shape(y_shape);

    // keep the front batch dimension same
    for (int i = 0; i < max_dim - 2; ++i) {
      if (new_x_shape[i] == new_y_shape[i]) {
        continue;
      }

      CHECK(new_x_shape[i] == 1 || new_y_shape[i] == 1)
          << "Input X and Y's batch dimension should be same or 1. But here " << matmul_info();

      // broadcast the value 1 dimension
      if (new_x_shape[i] == 1) {
        new_x_shape[i] = new_y_shape[i];
      } else {
        new_y_shape[i] = new_x_shape[i];
      }
    }
  }

  auto broad_x = x, broad_y = y;
  if (!new_x_shape.empty() && new_x_shape != x_shape) {
    int new_size = std::accumulate(new_x_shape.begin(), new_x_shape.end(), 1, std::multiplies<int>());
    int old_size = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int>());

    if (new_size == old_size) {
      VLOG(4) << "Reshape matmul's input X from [" << cinn::utils::Join(x_shape, ", ") << "] to ["
              << cinn::utils::Join(new_x_shape, ", ") << "]. Where " << matmul_info();
      broad_x = Reshape(x, new_x_shape);
    } else {
      VLOG(4) << "Broadcast matmul's input X from [" << cinn::utils::Join(x_shape, ", ") << "] to ["
              << cinn::utils::Join(new_x_shape, ", ") << "]. Where " << matmul_info();
      broad_x = BroadcastTo(x, new_x_shape);
    }
  }

  if (!new_y_shape.empty() && new_y_shape != y_shape) {
    int new_size = std::accumulate(new_y_shape.begin(), new_y_shape.end(), 1, std::multiplies<int>());
    int old_size = std::accumulate(y_shape.begin(), y_shape.end(), 1, std::multiplies<int>());

    if (new_size == old_size) {
      // only need reshape
      VLOG(4) << "Reshape matmul's input Y from [" << cinn::utils::Join(y_shape, ", ") << "] to ["
              << cinn::utils::Join(new_y_shape, ", ") << "]. Where " << matmul_info();
      broad_y = Reshape(y, new_y_shape);
    } else {
      // need broadcast
      VLOG(4) << "Broadcast matmul's input Y from [" << cinn::utils::Join(y_shape, ", ") << "] to ["
              << cinn::utils::Join(new_y_shape, ", ") << "]. Where " << matmul_info();
      broad_y = BroadcastTo(y, new_y_shape);
    }
  }

  return {broad_x, broad_y};
}

std::vector<int> NetBuilder::GetMatmulOutputShape(
    const Variable& x, const Variable& y, bool trans_x, bool trans_y, float alpha) {
  const auto &x_shape = x->shape, &y_shape = y->shape;

  auto matmul_info = [&]() {
    std::stringstream ss;
    ss << "matmul(X:" << x->id << "[" << cinn::utils::Join(x_shape, ", ") << "], Y:" << y->id << "["
       << cinn::utils::Join(y_shape, ", ") << "]"
       << ", trans_x=" << trans_x << ", trans_y=" << trans_y << ", alpha=" << alpha << ")";
    return ss.str();
  };

  int x_dim = x_shape.size(), y_dim = y_shape.size();
  int max_dim = std::max(x_shape.size(), y_shape.size());

  std::vector<int> out_shape;
  if (max_dim == 1) {
    // vector * vector
    CHECK(x_shape == y_shape)
        << "The matmul input X's numbers must be equal to Y's numbers,when X/Y's dims =1. But here " << matmul_info();

    out_shape = {1};
  } else if (x_dim == 1) {
    // vector * matrix
    out_shape = y_shape;
    if (trans_y) {
      // [m] * [a, b, d, m] -> [a, b, d]
      out_shape.erase(out_shape.end() - 1);
    } else {
      // [m] * [a, b, m, d] -> [a, b, d]
      out_shape.erase(out_shape.end() - 2);
    }
  } else if (y_dim == 1) {
    // matrix * vector
    out_shape = x_shape;
    if (trans_x) {
      // [a, b, m, c] * [m] -> [a, b, c]
      out_shape.erase(out_shape.end() - 2);
    } else {
      // [a, b, c, m] * [m] -> [a, b, c]
      out_shape.erase(out_shape.end() - 1);
    }
  } else {
    // matrix * matrix
    int M = trans_x ? x_shape[x_dim - 1] : x_shape[x_dim - 2];
    int N = trans_y ? y_shape[y_dim - 2] : y_shape[y_dim - 1];

    out_shape.resize(max_dim, 1);
    out_shape[max_dim - 2] = M;
    out_shape[max_dim - 1] = N;

    // get the batch dimension after broadcast
    int x_pos = x_dim - 3, y_pos = y_dim - 3, out_pos = max_dim - 3;
    while (x_pos >= 0 && y_pos >= 0) {
      CHECK(x_shape[x_pos] == y_shape[y_pos] || x_shape[x_pos] == 1 || y_shape[y_pos] == 1)
          << "Input X and Y's batch dimension should be same or 1. But here " << matmul_info();
      out_shape[out_pos] = (x_shape[x_pos] == 1) ? y_shape[y_pos] : x_shape[x_pos];

      out_pos--;
      x_pos--;
      y_pos--;
    }

    while (x_pos >= 0) {
      out_shape[out_pos--] = x_shape[x_pos--];
    }
    while (y_pos >= 0) {
      out_shape[out_pos--] = x_shape[y_pos--];
    }
  }
  return out_shape;
}

Variable NetBuilder::Matmul(const Variable& x, const Variable& y, bool trans_x, bool trans_y, float alpha) {
  const auto& inputs = BroadcastMatmulInput(x, y, trans_x, trans_y, alpha);

  auto out =
      CustomInstr(
          "matmul", {inputs.first, inputs.second}, {{"trans_a", trans_x}, {"trans_b", trans_y}, {"alpha", alpha}})
          .front();

  const auto& should_out_shape = GetMatmulOutputShape(x, y, trans_x, trans_y, alpha);
  if (should_out_shape != out->shape) {
    int should_out_size = std::accumulate(should_out_shape.begin(), should_out_shape.end(), 1, std::multiplies<int>());
    int real_out_size   = std::accumulate(out->shape.begin(), out->shape.end(), 1, std::multiplies<int>());
    CHECK_EQ(should_out_size, real_out_size)
        << "Cannot reshape the output:[" << out->id << "] of matmul from [" << cinn::utils::Join(out->shape, ", ")
        << "] to [" << cinn::utils::Join(should_out_shape, ", ") << "]."
        << " Whose input is "
        << "matmul(X:" << x->id << "[" << cinn::utils::Join(x->shape, ", ") << "], Y:" << y->id << "["
        << cinn::utils::Join(y->shape, ", ") << "]"
        << ", trans_x=" << trans_x << ", trans_y=" << trans_y << ", alpha=" << alpha << ")";
    out = Reshape(out, should_out_shape);
  }

  return out;
}

}  // namespace frontend
}  // namespace cinn

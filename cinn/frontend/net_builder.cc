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

Variable NetBuilder::ElementwiseOp(const std::string& op_type, const Variable& lhs, const Variable& rhs, int axis) {
  Instruction instr(op_type, {lhs, rhs});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::UnaryOp(const std::string& op_type, const Variable& operand) {
  Instruction instr(op_type, {operand});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs) {
  Instruction instr(op_type, {lhs, rhs});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

#define NETBUILDER_UNARY_OP_DEF(func_name__, op_type__) \
  Variable NetBuilder::func_name__(const Variable& operand) { return UnaryOp(#op_type__, operand); }
NETBUILDER_UNARY_OP_DEF(Sqrt, sqrt)
NETBUILDER_UNARY_OP_DEF(Tanh, tanh)
NETBUILDER_UNARY_OP_DEF(Relu, relu)
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

#define NETBUILDER_BINARY_OP_DEF(func_name__, op_type__) \
  Variable NetBuilder::func_name__(const Variable& lhs, const Variable& rhs) { return BinaryOp(#op_type__, lhs, rhs); }
NETBUILDER_BINARY_OP_DEF(Add, elementwise_add)
NETBUILDER_BINARY_OP_DEF(Sub, substract)
NETBUILDER_BINARY_OP_DEF(Div, divide)
NETBUILDER_BINARY_OP_DEF(ReluGrad, relu_grad)
NETBUILDER_BINARY_OP_DEF(Dot, matmul)
NETBUILDER_BINARY_OP_DEF(FloorDiv, floor_divide)
NETBUILDER_BINARY_OP_DEF(Mod, mod)
NETBUILDER_BINARY_OP_DEF(FloorMod, floor_mod)
NETBUILDER_BINARY_OP_DEF(Max, max)
NETBUILDER_BINARY_OP_DEF(Min, min)
NETBUILDER_BINARY_OP_DEF(Power, power)
NETBUILDER_BINARY_OP_DEF(LogicalAnd, logical_and)
NETBUILDER_BINARY_OP_DEF(LogicalOr, logical_or)
NETBUILDER_BINARY_OP_DEF(LogicalXor, logical_xor)
NETBUILDER_BINARY_OP_DEF(BitwiseAnd, bitwise_and)
NETBUILDER_BINARY_OP_DEF(BitwiseOr, bitwise_or)
NETBUILDER_BINARY_OP_DEF(BitwiseXor, bitwise_xor)
NETBUILDER_BINARY_OP_DEF(LeftShift, left_shift)
NETBUILDER_BINARY_OP_DEF(RightShift, right_shift)

#define NETBUILDER_ELEMENTWISE_OP_DEF(func_name__, op_type__)                            \
  Variable NetBuilder::func_name__(const Variable& lhs, const Variable& rhs, int axis) { \
    return ElementwiseOp(#op_type__, lhs, rhs, axis);                                    \
  }
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseAdd, elementwise_add)
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseMul, elementwise_mul)
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseDiv, divide)
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseSub, substract)

const std::vector<Variable>& NetBuilder::CustomInstr(const std::string& type,
                                                     const std::vector<Variable>& inputs,
                                                     const AttributeMap& attrs) {
  Instruction instr(type);
  instr.SetInputs(inputs);
  for (auto& kv : attrs) {
    instr.SetAttr(kv.first, kv.second);
  }

  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable NetBuilder::Compare(const Variable& lhs, const Variable& rhs, ComparisonKind kind) {
  switch (kind) {
    case ComparisonKind::kEq:
      return BinaryOp("equal", lhs, rhs);
    case ComparisonKind::kNe:
      return BinaryOp("not_equal", lhs, rhs);
    case ComparisonKind::kGe:
      return BinaryOp("greater_equal", lhs, rhs);
    case ComparisonKind::kGt:
      return BinaryOp("greater", lhs, rhs);
    case ComparisonKind::kLe:
      return BinaryOp("less_equal", lhs, rhs);
    case ComparisonKind::kLt:
      return BinaryOp("less", lhs, rhs);
    default:
      LOG(FATAL) << "unknown comparison kind";
  }
}

std::vector<Variable> NetBuilder::Split(const Variable& operand, const std::vector<int>& num_or_sections, int axis) {
  Instruction instr("split", {operand});
  instr.SetAttr("num_or_sections", num_or_sections);
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable NetBuilder::Concat(const std::vector<Variable>& input_vars, int axis) {
  Instruction instr("concat", input_vars);
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Reduce(const Variable& operand, ReduceKind kind, const std::vector<int>& dim, bool keep_dim) {
  auto reduce_func = [&](const std::string& op_type) {
    Instruction instr(op_type, {operand});
    instr.SetAttr("dim", dim);
    instr.SetAttr("keep_dim", keep_dim);
    InferShape(instr);
    AppendInstruction(instr);
    return instr.GetOutput(0);
  };

  switch (kind) {
    case ReduceKind::kSum:
      return reduce_func("reduce_sum");
    case ReduceKind::kProd:
      return reduce_func("reduce_prod");
    case ReduceKind::kMax:
      return reduce_func("reduce_max");
    case ReduceKind::kMin:
      return reduce_func("reduce_min");
    case ReduceKind::kAll:
      return reduce_func("reduce_all");
    case ReduceKind::kAny:
      return reduce_func("reduce_any");
    default:
      LOG(FATAL) << "unknown reduction kind";
  }
}

Variable NetBuilder::FillConstant(
    const std::vector<int>& shape, float value, const std::string& name, const std::string& dtype, bool force_cpu) {
  Instruction instr("fill_constant");
  instr.SetInputs({});
  instr.SetAttr("shape", shape);
  instr.SetAttr("value", value);
  instr.SetAttr("dtype", dtype);
  instr.SetAttr("force_cpu", force_cpu);

  InferShape(instr);
  AppendInstruction(instr);
  auto out = instr.GetOutput(0);
  out.set_id(name);
  return out;
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
  Instruction instr("broadcast_to", {operand});
  instr.SetAttr("out_shape", out_shape);
  instr.SetAttr("broadcast_axes", broadcast_axes);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Reshape(const Variable& operand, const std::vector<int>& shape) {
  Instruction instr("reshape", {operand});
  instr.SetAttr("shape", shape);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Transpose(const Variable& operand, const std::vector<int>& axis) {
  Instruction instr("transpose", {operand});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Slice(const Variable& operand,
                           const std::vector<int>& axes,
                           const std::vector<int>& starts,
                           const std::vector<int>& ends,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& strides) {
  Instruction instr("slice", {operand});
  instr.SetAttr("axes", axes);
  instr.SetAttr("starts", starts);
  instr.SetAttr("ends", ends);
  instr.SetAttr("infer_flags", infer_flags);
  instr.SetAttr("strides", strides);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::SliceAssign(const Variable& input,
                                 const Variable& assign,
                                 const std::vector<int>& axes,
                                 const std::vector<int>& starts,
                                 const std::vector<int>& ends,
                                 const std::vector<int>& strides) {
  Instruction instr("slice_assign", {input, assign});
  instr.SetAttr("axes", axes);
  instr.SetAttr("starts", starts);
  instr.SetAttr("ends", ends);
  instr.SetAttr("strides", strides);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Reverse(const Variable& operand, const std::vector<int>& axis) {
  Instruction instr("reverse", {operand});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Select(const Variable& condition, const Variable& true_value, const Variable& false_value) {
  Instruction instr("select", {condition, true_value, false_value});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::IndexSelect(const Variable& operand, const Variable& index, int axis) {
  Instruction instr("index_select", {operand, index});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::ScatterAssign(const Variable& operand, const Variable& updates, const Variable& index, int axis) {
  Instruction instr("scatter_assign", {operand, updates, index});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::ScatterAdd(const Variable& operand, const Variable& updates, const Variable& index, int axis) {
  Instruction instr("scatter_add", {operand, updates, index});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::IsClose(const Variable& x, const Variable& y, float rtol, float atol, bool equal_nan) {
  Instruction instr("isclose", {x, y});
  instr.SetAttr("rtol", rtol);
  instr.SetAttr("atol", atol);
  instr.SetAttr("equal_nan", equal_nan);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Mul(const Variable& a, const Variable& b, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mul", {a, b});
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

const std::vector<Variable>& NetBuilder::ElementwiseAddGrad(const Variable& dout,
                                                            const Variable& x,
                                                            const Variable& y,
                                                            int axis) {
  Instruction instr("elementwise_add_grad", {dout, x, y});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable NetBuilder::Relu6(const Variable& a, float threshold) {
  Instruction instr("relu6", {a});
  instr.SetAttr("threshold", threshold);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::ReduceSum(const Variable& x, const std::vector<int>& dim, bool keep_dim) {
  return Reduce(x, ReduceKind::kSum, dim, keep_dim);
}

Variable NetBuilder::ReduceAll(const Variable& x, const std::vector<int>& dim, bool keep_dim) {
  return Reduce(x, ReduceKind::kAll, dim, keep_dim);
}

Variable NetBuilder::ReduceAny(const Variable& x, const std::vector<int>& dim, bool keep_dim) {
  return Reduce(x, ReduceKind::kAny, dim, keep_dim);
}

Variable NetBuilder::Cast(const Variable& operand, const std::string& dtype) {
  Instruction instr("cast", {operand});
  instr.SetAttr("dtype", dtype);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Squeeze(const Variable& operand, const std::vector<int>& axes) {
  Instruction instr("squeeze", {operand});
  instr.SetAttr("axes", axes);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
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
  Instruction instr("conv2d");
  instr.SetInputs({lhs, rhs});
  instr.SetAttr("stride", strides);
  instr.SetAttr("padding", paddings);
  instr.SetAttr("dilation", dilations);
  instr.SetAttr("groups", groups);
  instr.SetAttr("conv_type", conv_type);
  instr.SetAttr("data_format", data_format);
  instr.SetAttr("padding_algorithm", padding_algorithm);
  instr.SetAttr("output_shape", output_shape);

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
  Instruction instr("conv2d");
  instr.SetInputs({a, b});
  instr.SetAttr("stride", strides);
  instr.SetAttr("padding", paddings);
  instr.SetAttr("dilation", dilations);
  instr.SetAttr("groups", groups);
  instr.SetAttr("data_format", data_format);
  instr.SetAttr("padding_algorithm", padding_algorithm);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::DepthwiseConv2d(const Variable& a,
                                     const Variable& b,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings,
                                     const std::vector<int>& dilations,
                                     int groups,
                                     const std::string& data_format,
                                     const std::string& padding_algorithm) {
  Instruction instr("depthwise_conv2d");
  instr.SetInputs({a, b});
  instr.SetAttr("stride", strides);
  instr.SetAttr("padding", paddings);
  instr.SetAttr("dilation", dilations);
  instr.SetAttr("groups", groups);
  instr.SetAttr("data_format", data_format);
  instr.SetAttr("padding_algorithm", padding_algorithm);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
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
  Instruction instr("pool2d");
  instr.SetInputs({a});
  instr.SetAttr("pool_type", pooling_type);
  instr.SetAttr("kernel_size", ksize);
  instr.SetAttr("stride_size", strides);
  instr.SetAttr("padding_size", paddings);
  instr.SetAttr("ceil_mode", ceil_mode);
  instr.SetAttr("exclusive", exclusive);
  instr.SetAttr("global_pooling", global_pooling);
  instr.SetAttr("data_format", data_format);
  instr.SetAttr("adaptive", adaptive);
  instr.SetAttr("padding_algorithm", padding_algorithm);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
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
  std::unique_ptr<Instruction> instr;
  if (is_test) {
    instr = std::make_unique<Instruction>("batchnorm");
  } else {
    instr = std::make_unique<Instruction>("batch_norm_train");
  }
  instr->SetInputs({a, scale, bias, mean, variance});
  instr->SetAttr("epsilon", epsilon);
  instr->SetAttr("momentum", momentum);
  instr->SetAttr("data_layout", data_layout);
  InferShape(*instr);
  AppendInstruction(*instr);
  return instr->GetOutputs();
}

// batch norm grad, output(grad_x, grad_scale, grad_bias)
std::vector<Variable> NetBuilder::BatchNormGrad(const Variable& dy,
                                                const Variable& x,
                                                const Variable& scale,
                                                const Variable& save_mean,
                                                const Variable& save_variance,
                                                const float epsilon,
                                                const std::string& data_layout) {
  Instruction instr("batch_norm_grad", {dy, x, scale, save_mean, save_variance});
  instr.SetAttr("epsilon", epsilon);
  instr.SetAttr("data_layout", data_layout);

  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable NetBuilder::Scale(const Variable& a, float scale, float bias, bool bias_after_scale) {
  Instruction instr("scale", {a});
  instr.SetAttr("scale", scale);
  instr.SetAttr("bias", bias);
  instr.SetAttr("bias_after_scale", bias_after_scale);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Softmax(const Variable& a, int axis, const std::string& data_format) {
  Instruction instr("softmax", {a});
  instr.SetAttr("axis", axis);
  instr.SetAttr("data_format", data_format);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::DropoutInfer(const Variable& a, float dropout_prob, const std::string& dropout_implementation) {
  Instruction instr("dropout_infer", {a});
  instr.SetAttr("dropout_prob", dropout_prob);
  instr.SetAttr("dropout_implementation", dropout_implementation);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Sum(const std::vector<Variable>& inputs) {
  Instruction instr("sum", inputs);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::Clip(const std::vector<Variable>& inputs, const float& max_val, const float& min_val) {
  Instruction instr("clip", inputs);
  instr.SetAttr("max_val", max_val);
  instr.SetAttr("min_val", min_val);
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
  Instruction instr("conv2d_grad", {dy, x, w});
  instr.SetAttr<std::vector<int>>("strides", strides);
  instr.SetAttr<std::vector<int>>("paddings", paddings);
  instr.SetAttr<std::vector<int>>("dilations", dilations);
  instr.SetAttr<int>("groups", groups);
  instr.SetAttr<std::string>("data_format", data_format);
  instr.SetAttr<std::string>("padding_algorithm", padding_algorithm);

  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
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

  Instruction instr("matmul", {inputs.first, inputs.second});
  instr.SetAttr("trans_a", trans_x);
  instr.SetAttr("trans_b", trans_y);
  instr.SetAttr("alpha", alpha);
  InferShape(instr);
  AppendInstruction(instr);
  auto out = instr.GetOutput(0);

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

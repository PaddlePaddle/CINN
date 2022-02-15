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
NETBUILDER_UNARY_OP_DEF(sqrt, sqrt)
NETBUILDER_UNARY_OP_DEF(tanh, tanh)
NETBUILDER_UNARY_OP_DEF(relu, relu)
NETBUILDER_UNARY_OP_DEF(sigmoid, sigmoid)
NETBUILDER_UNARY_OP_DEF(identity, identity)

#define NETBUILDER_BINARY_OP_DEF(func_name__, op_type__) \
  Variable NetBuilder::func_name__(const Variable& lhs, const Variable& rhs) { return BinaryOp(#op_type__, lhs, rhs); }
NETBUILDER_BINARY_OP_DEF(sub, substract)
NETBUILDER_BINARY_OP_DEF(div, divide)
NETBUILDER_BINARY_OP_DEF(matmul, matmul)
NETBUILDER_BINARY_OP_DEF(relu_grad, relu_grad)

Variable NetBuilder::add(const Variable& a, const Variable& b) {
  Instruction instr("elementwise_add", {a, b});
  instr.SetAttr("axis", -1);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::reshape(const Variable& operand, const std::vector<int>& shape) {
  Instruction instr("reshape", {operand});
  instr.SetAttr("shape", shape);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::transpose(const Variable& operand, const std::vector<int>& axis) {
  Instruction instr("transpose", {operand});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::concat(const std::vector<Variable>& inputs, int axis) {
  Instruction instr("concat", inputs);
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::mul(const Variable& a, const Variable& b, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mul", {a, b});
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::mulbias(
    const Variable& a, const Variable& b, const Variable& c, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mulbias", {a, b, c});
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(1);
}

Variable NetBuilder::elementwise_add(const Variable& a, const Variable& b, int axis) {
  Instruction instr("elementwise_add", {a, b});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

const std::vector<Variable>& NetBuilder::elementwise_add_grad(const Variable& dout,
                                                              const Variable& x,
                                                              const Variable& y,
                                                              int axis) {
  Instruction instr("elementwise_add_grad", {dout, x, y});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable NetBuilder::elementwise_mul(const Variable& a, const Variable& b, int axis) {
  Instruction instr("elementwise_mul", {a, b});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::relu6(const Variable& a, float threshold) {
  Instruction instr("relu6", {a});
  instr.SetAttr("threshold", threshold);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::reverse(const Variable& x, const std::vector<int>& axis) {
  Instruction instr("reverse", {x});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::reduce_sum(const Variable& x, const std::vector<int>& dim, bool keep_dim) {
  Instruction instr("reduce_sum", {x});
  instr.SetAttr("dim", dim);
  instr.SetAttr("keep_dim", keep_dim);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::conv2d(const Variable& a,
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

Variable NetBuilder::depthwise_conv2d(const Variable& a,
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

Variable NetBuilder::pool2d(const Variable& a,
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

std::vector<Variable> NetBuilder::batchnorm(const Variable& a,
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
std::vector<Variable> NetBuilder::batch_norm_grad(const Variable& dy,
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

Variable NetBuilder::scale(const Variable& a, float scale, float bias, bool bias_after_scale) {
  Instruction instr("scale", {a});
  instr.SetAttr("scale", scale);
  instr.SetAttr("bias", bias);
  instr.SetAttr("bias_after_scale", bias_after_scale);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::softmax(const Variable& a, int axis, const std::string& data_format) {
  Instruction instr("softmax", {a});
  instr.SetAttr("axis", axis);
  instr.SetAttr("data_format", data_format);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::slice(const Variable& a,
                           const std::vector<int>& axes,
                           const std::vector<int>& starts,
                           const std::vector<int>& ends,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& decrease_axis) {
  Instruction instr("slice", {a});
  instr.SetAttr("axes", axes);
  instr.SetAttr("starts", starts);
  instr.SetAttr("ends", ends);
  instr.SetAttr("infer_flags", infer_flags);
  instr.SetAttr("decrease_axis", decrease_axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::dropout_infer(const Variable& a, float dropout_prob, const std::string& dropout_implementation) {
  Instruction instr("dropout_infer", {a});
  instr.SetAttr("dropout_prob", dropout_prob);
  instr.SetAttr("dropout_implementation", dropout_implementation);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::sum(const std::vector<Variable>& inputs) {
  Instruction instr("sum", inputs);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

// conv2d grad, output(grad_x, grad_w)
std::vector<Variable> NetBuilder::conv2d_grad(const Variable& dy,
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

Variable NetBuilder::reduce(const Variable& a, ReduceKind kind, const std::vector<int>& dim, bool keep_dim) {
  auto reduce_func = [&](const std::string& op_type) {
    Instruction instr(op_type, {a});
    std::vector<int> new_dim(dim);
    if (dim.empty()) {
      for (int i = 0; i < a->shape.size(); ++i) {
        new_dim.push_back(i);
      }
    }
    instr.SetAttr("dim", new_dim);
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
    default:
      LOG(FATAL) << "unknown reduction kind";
  }
}

Variable NetBuilder::broadcast_to(const Variable& a,
                                  const std::vector<int>& out_shape,
                                  const std::vector<int>& broadcast_axes) {
  Instruction instr("broadcast_to", {a});
  instr.SetAttr("out_shape", out_shape);
  instr.SetAttr("broadcast_axes", broadcast_axes);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

}  // namespace frontend
}  // namespace cinn

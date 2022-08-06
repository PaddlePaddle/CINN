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

#define NETBUILDER_UNARY_OP_DEF(func_name__, op_type__) \
  Variable NetBuilder::func_name__(const Variable& operand) { return UnaryOp(#op_type__, operand); }
NETBUILDER_UNARY_OP_DEF(Sqrt, sqrt)
NETBUILDER_UNARY_OP_DEF(Tanh, tanh)
NETBUILDER_UNARY_OP_DEF(Relu, relu)
NETBUILDER_UNARY_OP_DEF(Sigmoid, sigmoid)
NETBUILDER_UNARY_OP_DEF(Identity, identity)

#define NETBUILDER_BINARY_OP_DEF(func_name__, op_type__) \
  Variable NetBuilder::func_name__(const Variable& lhs, const Variable& rhs) { return BinaryOp(#op_type__, lhs, rhs); }
NETBUILDER_BINARY_OP_DEF(Add, elementwise_add)
NETBUILDER_BINARY_OP_DEF(Sub, substract)
NETBUILDER_BINARY_OP_DEF(Div, divide)
NETBUILDER_BINARY_OP_DEF(Matmul, matmul)
NETBUILDER_BINARY_OP_DEF(ReluGrad, relu_grad)

#define NETBUILDER_ELEMENTWISE_OP_DEF(func_name__, op_type__)                            \
  Variable NetBuilder::func_name__(const Variable& lhs, const Variable& rhs, int axis) { \
    return ElementwiseOp(#op_type__, lhs, rhs, axis);                                    \
  }
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseAdd, elementwise_add)
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseMul, elementwise_mul)

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

Variable NetBuilder::OneHot(const Variable& indices,
                            const Variable& on_value,
                            const Variable& off_value,
                            const int depth,
                            const int axis,
                            const std::string& dtype) {
  Instruction instr("one_hot");
  instr.SetInputs({indices, on_value, off_value});
  instr.SetAttr("depth", depth);
  instr.SetAttr("axis", axis);
  instr.SetAttr("dtype", dtype);
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

Variable NetBuilder::ElementwiseOp(const std::string& op_type, const Variable& lhs, const Variable& rhs, int axis) {
  Instruction instr(op_type, {lhs, rhs});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

}  // namespace frontend
}  // namespace cinn

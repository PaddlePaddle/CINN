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
Variable NetBuilder::add(const Variable& a, const Variable& b) {
  Instruction instr("elementwise_add", {a, b});
  instr.SetAttr("axis", -1);
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

Variable NetBuilder::relu(const Variable& a) {
  Instruction instr("relu", {a});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable NetBuilder::relu_grad(const Variable& dout, const Variable& out) {
  Instruction instr("relu_grad", {dout, out});
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
  instr.SetAttr("pooling_type", pooling_type);
  instr.SetAttr("ksize", ksize);
  instr.SetAttr("strides", strides);
  instr.SetAttr("paddings", paddings);
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

Variable NetBuilder::sigmoid(const Variable& a) {
  Instruction instr("sigmoid", {a});
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

}  // namespace frontend
}  // namespace cinn

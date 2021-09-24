#include "cinn/frontend/symbolization/coarse_builder.h"

#include <string>
#include <unordered_map>
#include <utility>

#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace symbolization {

Variable CoarseBuilder::conv2d(const Variable& a,
                               const Variable& b,
                               const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::layout_transform(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("layout_transform");
  instr.SetInputs({a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::conv2d_NCHWc(const Variable& a,
                                     const Variable& b,
                                     const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("conv2d_NCHWc");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::depthwise_conv2d(const Variable& a,
                                         const Variable& b,
                                         const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("depthwise_conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::pool2d(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("pool2d");
  instr.SetInputs({a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::batchnorm(const Variable& a,
                                  const Variable& scale,
                                  const Variable& bias,
                                  const Variable& mean,
                                  const Variable& variance,
                                  const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("batchnorm");
  instr.SetInputs({a, scale, bias, mean, variance});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::scale(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("scale", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::softmax(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("softmax", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::sigmoid(const Variable& a) {
  Instruction instr("sigmoid", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::slice(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("slice", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::dropout_infer(const Variable& a, const std::unordered_map<std::string, AttrT>& attr_store) {
  Instruction instr("dropout_infer", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::add(const Variable& a, const Variable& b) {
  Instruction instr("elementwise_add", {a, b});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::elementwise_add(const Variable& a, const Variable& b, int axis) {
  Instruction instr("elementwise_add", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::elementwise_mul(const Variable& a, const Variable& b, int axis) {
  Instruction instr("elementwise_mul", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::relu(const Variable& a) {
  Instruction instr("relu", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::relu6(const Variable& a) {
  Instruction instr("relu6", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::mul(const Variable& a, const Variable& b, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mul", {a, b});
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CoarseBuilder::mulbias(
    const Variable& a, const Variable& b, const Variable& c, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mulbias", {a, b, c});
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  AppendInstruction(instr);
  return instr.GetOutput(1);
}

}  // namespace symbolization
}  // namespace frontend
}  // namespace cinn

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
NETBUILDER_BINARY_OP_DEF(ReluGrad, relu_grad)

#define NETBUILDER_ELEMENTWISE_OP_DEF(func_name__, op_type__)                            \
  Variable NetBuilder::func_name__(const Variable& lhs, const Variable& rhs, int axis) { \
    return ElementwiseOp(#op_type__, lhs, rhs, axis);                                    \
  }
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseAdd, elementwise_add)
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseMul, elementwise_mul)
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseDiv, divide)
NETBUILDER_ELEMENTWISE_OP_DEF(ElementwiseSub, substract)

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

Variable NetBuilder::Flip(const Variable& inputs, const std::vector<int>& axis) {
  Instruction instr("Flip", {inputs});
  instr.SetAttr("axis", axis);
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

Variable NetBuilder::ElementwiseOp(const std::string& op_type, const Variable& lhs, const Variable& rhs, int axis) {
  Instruction instr(op_type, {lhs, rhs});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

}  // namespace frontend
}  // namespace cinn

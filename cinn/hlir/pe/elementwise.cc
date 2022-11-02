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

#include "cinn/hlir/pe/elementwise.h"

#include <string>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/builtin.h"

namespace cinn {
namespace hlir {
namespace pe {

using ir::Expr;
using ir::Tensor;
using lang::Compute;

#define HLIR_IMP_UNARY_PE(name__)                                                                          \
  std::vector<ir::Tensor> name__(const Tensor& A, const std::string& output_name) {                        \
    return {Compute(                                                                                       \
        A->shape, [=](const std::vector<Expr>& indice) { return lang::name__(A(indice)); }, output_name)}; \
  }

#define HLIR_MKL_IMP_UNARY_PE(name__, ex_name__)                                                     \
  std::vector<ir::Tensor> name__##MKL(const Tensor& A, const std::string& output_name) {             \
    CHECK(A->type().is_float()) << "type should be float or double but get " << A->type();           \
    std::string fn_name = "cinn_mkl_" #ex_name__ "_v_fp" + std::to_string(A->type().bits());         \
    auto call           = Compute(                                                                   \
        {Expr(1)}, [=]() -> Expr { return lang::CallExtern(fn_name, {A}); }, output_name); \
    auto out = call->TupleGet(0);                                                                    \
    out->WithBuffer(A->type());                                                                      \
    return {out, call};                                                                              \
  }

HLIR_MKL_IMP_UNARY_PE(Exp, exp);
HLIR_MKL_IMP_UNARY_PE(Erf, erf);
HLIR_MKL_IMP_UNARY_PE(Sqrt, sqrt);
HLIR_MKL_IMP_UNARY_PE(Log, log);
HLIR_MKL_IMP_UNARY_PE(Log2, log2);
HLIR_MKL_IMP_UNARY_PE(Log10, log10);
HLIR_MKL_IMP_UNARY_PE(Floor, floor);
HLIR_MKL_IMP_UNARY_PE(Ceil, ceil);
HLIR_MKL_IMP_UNARY_PE(Round, round);
HLIR_MKL_IMP_UNARY_PE(Tanh, tanh);
HLIR_MKL_IMP_UNARY_PE(Trunc, trunc);
HLIR_MKL_IMP_UNARY_PE(Cos, cos);
HLIR_MKL_IMP_UNARY_PE(Sin, sin);
HLIR_MKL_IMP_UNARY_PE(Cosh, cosh);
HLIR_MKL_IMP_UNARY_PE(Tan, tan);
HLIR_MKL_IMP_UNARY_PE(Sinh, sinh);
HLIR_MKL_IMP_UNARY_PE(Acos, acos);
HLIR_MKL_IMP_UNARY_PE(Acosh, acosh);
HLIR_MKL_IMP_UNARY_PE(Asin, asin);
HLIR_MKL_IMP_UNARY_PE(Asinh, asinh);
HLIR_MKL_IMP_UNARY_PE(Atan, atan);
HLIR_MKL_IMP_UNARY_PE(Atanh, atanh);

HLIR_IMP_UNARY_PE(Exp);
HLIR_IMP_UNARY_PE(Erf);
HLIR_IMP_UNARY_PE(Sqrt);
HLIR_IMP_UNARY_PE(Log);
HLIR_IMP_UNARY_PE(Log2);
HLIR_IMP_UNARY_PE(Log10);
HLIR_IMP_UNARY_PE(Floor);
HLIR_IMP_UNARY_PE(Ceil);
HLIR_IMP_UNARY_PE(Round);
HLIR_IMP_UNARY_PE(Trunc);
HLIR_IMP_UNARY_PE(Cos);
HLIR_IMP_UNARY_PE(Cosh);
HLIR_IMP_UNARY_PE(Tan);
HLIR_IMP_UNARY_PE(Sin);
HLIR_IMP_UNARY_PE(Sinh);
HLIR_IMP_UNARY_PE(Acos);
HLIR_IMP_UNARY_PE(Acosh);
HLIR_IMP_UNARY_PE(Asin);
HLIR_IMP_UNARY_PE(Asinh);
HLIR_IMP_UNARY_PE(Atan);
HLIR_IMP_UNARY_PE(Atanh);
HLIR_IMP_UNARY_PE(IsNan);
HLIR_IMP_UNARY_PE(Tanh);
HLIR_IMP_UNARY_PE(IsFinite);
HLIR_IMP_UNARY_PE(IsInf);

HLIR_IMP_UNARY_PE(Negative);
HLIR_IMP_UNARY_PE(Identity);
HLIR_IMP_UNARY_PE(LogicalNot);
HLIR_IMP_UNARY_PE(BitwiseNot);
HLIR_IMP_UNARY_PE(Sigmoid);
HLIR_IMP_UNARY_PE(Sign);
HLIR_IMP_UNARY_PE(Abs);
HLIR_IMP_UNARY_PE(Rsqrt);

ir::Tensor Reshape(const ir::Tensor& A, const std::vector<int>& new_shape, const std::string& name) {
  std::vector<Expr> new_expr_shape;
  std::vector<Expr> A_expr_shape = A->shape;
  int input_total_size           = 1;
  int output_total_size          = 1;
  for (auto& i : A_expr_shape) {
    CHECK(i.is_constant()) << "Input tensor's shape should be constant value.";
    input_total_size *= static_cast<int>(i.get_constant());
  }
  for (auto& i : new_shape) {
    output_total_size *= i;
    new_expr_shape.push_back(Expr(i));
  }
  CHECK_EQ(input_total_size, output_total_size)
      << "In op reshape, the input tensor and output tensor's total size should be equal, please check!";
  auto res = Compute(
      new_expr_shape,
      [=](const std::vector<Expr>& indice) {
        Expr offset = Expr(0);
        for (int i = 0; i < indice.size(); i++) {
          offset = offset * new_expr_shape[i] + indice[i];
        }
        std::vector<Expr> indice_a;
        for (int i = A_expr_shape.size() - 1; i >= 0; i--) {
          auto temp = offset % A_expr_shape[i];
          indice_a.insert(indice_a.begin(), temp);
          offset = (offset - temp) / A_expr_shape[i];
        }
        return A(indice_a);
      },
      name);
  return res;
}

ir::Tensor ExpandDims(const ir::Tensor& input, int axis, int num_newaxis, const std::string& output_name) {
  int ndims = input.ndims();
  CHECK(axis >= -ndims - 1 && axis <= ndims)
      << "expand_dims only accept `axis` in [-x.ndim - 1, x.ndim], but got axis = " << axis
      << ", and x.ndims = " << ndims;
  CHECK_GE(num_newaxis, 0);

  if (axis < 0) {
    axis = ndims + axis + 1;
  }
  std::vector<Expr> output_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    output_shape.push_back(input->shape[i]);
  }
  for (size_t i = 0; i < static_cast<size_t>(num_newaxis); ++i) {
    output_shape.push_back(Expr(1));
  }
  for (size_t i = axis; i < input->shape.size(); ++i) {
    output_shape.push_back(input->shape[i]);
  }

  return Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> idx;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
          idx.push_back(indice[i]);
        }
        for (size_t i = axis + num_newaxis; i < indice.size(); ++i) {
          idx.push_back(indice[i]);
        }
        return input(idx);
      },
      UniqName(output_name));
}

ir::Tensor Squeeze(const ir::Tensor& A, const std::vector<int>& axes, const std::string& output_name) {
  std::vector<int> position;
  std::vector<Expr> output_shape;
  if (axes.size()) {
    for (int idx = 0; idx < A->shape.size(); ++idx) {
      // if can't find idx in axis
      if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
        output_shape.push_back(A->shape[idx]);
        position.push_back(idx);
      } else {
        CHECK_EQ(A->shape[idx], Expr(1));
      }
    }
  } else {
    for (int idx = 0; idx < A->shape.size(); ++idx) {
      if (A->shape[idx] != Expr(1)) {
        output_shape.push_back(A->shape[idx]);
        position.push_back(idx);
      }
    }
  }

  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indices) {
        std::vector<Expr> indexs(A->shape.size(), Expr(0));
        for (int idx = 0; idx < indices.size(); ++idx) {
          indexs[position[idx]] = indices[idx];
        }
        return A(indexs);
      },
      output_name);
  return res;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

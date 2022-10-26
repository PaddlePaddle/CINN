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

#include "cinn/hlir/pe/broadcast.h"

#include <algorithm>

#include "cinn/common/ir_util.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace hlir {
namespace pe {

using common::make_zero;
using ir::Tensor;
using lang::Compute;

void GetBroadcastShape(const std::vector<Expr>& shape1,
                       const std::vector<Expr>& shape2,
                       std::vector<Expr>* common_shape,
                       std::vector<bool>* broadcast_flag1,
                       std::vector<bool>* broadcast_flag2,
                       int* axis_offset,
                       const Expr& axis) {
  CHECK(common_shape);
  CHECK(broadcast_flag1);
  CHECK(broadcast_flag2);

  std::vector<Expr> shape1_new = shape1;
  std::vector<Expr> shape2_new = shape2;

  if (axis.defined()) {
    int axis_val = axis.as_int32();
    CHECK_GE(axis_val, -1) << "wrong axis: " << axis_val << std::endl;
    if (shape1.size() >= shape2.size()) {
      CHECK_LE(axis_val, static_cast<int>(shape1.size() - shape2.size()))
          << "wrong axis: " << axis_val << " is not <= " << shape1.size() - shape2.size() << std::endl;
      if (axis_val >= 0) {
        *axis_offset = shape1.size() - shape2.size() - axis_val;
        for (int i = 1; i <= *axis_offset; ++i) {
          // specified axis to align, we insert Expr one in tensor B so as to align right with tensor A.
          shape2_new.emplace_back(Expr(1));
          common_shape->insert(common_shape->begin(), shape1[static_cast<int>(shape1.size() - i)]);
          // flag is used to indicate whether to include the indice or not.
          broadcast_flag1->emplace_back(true);
          broadcast_flag2->emplace_back(false);
        }
      }
    } else {
      CHECK_LE(axis_val, static_cast<int>(shape2.size() - shape1.size()))
          << "wrong axis: " << axis_val << " is not <= " << shape2.size() - shape1.size() << std::endl;
      if (axis_val >= 0) {
        *axis_offset = shape2.size() - shape1.size() - axis_val;
        for (int i = 1; i <= *axis_offset; ++i) {
          // specified axis to align, we insert Expr one in tensor B so as to align right with tensor A.
          shape1_new.emplace_back(Expr(1));
          common_shape->insert(common_shape->begin(), shape2[static_cast<int>(shape2.size() - i)]);
          // flag is used to indicate whether to include the indice or not.
          broadcast_flag2->emplace_back(true);
          broadcast_flag1->emplace_back(false);
        }
      }
    }
  }

  int size1 = shape1_new.size();
  int size2 = shape2_new.size();

  Expr one(1);
  int i;
  i = axis_offset <= 0 ? 1 : *axis_offset + 1;
  for (; i <= std::min(size1, size2); ++i) {
    // traverse from right to left to get the output shape and broadcast flag
    auto* var1 = shape1_new[size1 - i].As<ir::_Var_>();
    auto* var2 = shape2_new[size2 - i].As<ir::_Var_>();
    if (MathEqual(shape1_new[size1 - i], shape2_new[size2 - i])) {
      common_shape->insert(common_shape->begin(), shape1_new[size1 - i]);
      // broadcast flags are recorded in a reverse order
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (MathEqual(one, shape1_new[size1 - i])) {
      CHECK(!MathEqual(one, shape2_new[size2 - i]));
      common_shape->insert(common_shape->begin(), shape2_new[size2 - i]);
      broadcast_flag1->emplace_back(false);
      broadcast_flag2->emplace_back(true);
    } else if (MathEqual(one, shape2_new[size2 - i])) {
      CHECK(!MathEqual(one, shape1_new[size1 - i]));
      common_shape->insert(common_shape->begin(), shape1_new[size1 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(false);
    } else if (var1 && var2) {
      Expr max_var = ir::Max::Make(shape1_new[size1 - i], shape2_new[size2 - i]);
      common_shape->insert(common_shape->begin(), max_var);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (var1) {
      common_shape->insert(common_shape->begin(), shape2_new[size2 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (var2) {
      common_shape->insert(common_shape->begin(), shape1_new[size1 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else {
      LOG(FATAL) << "Incompatible broadcast dims " << shape1_new[size1 - i] << " and " << shape2_new[size2 - i]
                 << " in: " << shape1_new << " and " << shape2_new << std::endl;
    }
  }
  if (size1 != size2) {
    int max_size = std::max(size1, size2);
    auto& shape  = (size1 > size2) ? shape1_new : shape2_new;
    auto var_l   = (size1 > size2) ? broadcast_flag1 : broadcast_flag2;
    auto var_s   = (size1 > size2) ? broadcast_flag2 : broadcast_flag1;
    for (; i <= max_size; ++i) {
      common_shape->insert(common_shape->begin(), shape[max_size - i]);
      var_l->emplace_back(true);
      var_s->emplace_back(false);
    }
  }
}

void GetBroadcastOutShape(const std::vector<int>& input_shape1,
                          const std::vector<int>& input_shape2,
                          std::vector<int>* common_shape,
                          int axis) {
  std::vector<Expr> shape1;
  std::vector<Expr> shape2;
  auto fn_expr = [](const std::vector<int>& input_shape, std::vector<Expr>* shape) {
    for (int i = 0; i < input_shape.size(); i++) {
      shape->push_back(Expr(input_shape[i]));
    }
  };
  fn_expr(input_shape1, &shape1);
  fn_expr(input_shape2, &shape2);
  std::vector<bool> broadcast_flags1;
  std::vector<bool> broadcast_flags2;
  int axis_offset = 0;
  std::vector<Expr> out_shape;
  GetBroadcastShape(shape1, shape2, &out_shape, &broadcast_flags1, &broadcast_flags2, &axis_offset, Expr(axis));
  CHECK(common_shape);
  for (auto& shape : out_shape) {
    common_shape->push_back(shape.as_int32());
  }
}

void GetBroadcastIndice(const std::vector<Expr>& indice,
                        const Tensor& tensor_a,
                        const Tensor& tensor_b,
                        int axis_offset,
                        std::vector<Expr>* broadcast_indice1,
                        std::vector<Expr>* broadcast_indice2,
                        const std::vector<bool>& broadcast_flags1,
                        const std::vector<bool>& broadcast_flags2) {
  CHECK(broadcast_indice1);
  CHECK(broadcast_indice2);
  if (broadcast_indice1->empty() && broadcast_indice2->empty()) {
    int flag_size = broadcast_flags1.size();
    int i;
    CHECK_GE(indice.size(), flag_size);
    for (i = 0; i < flag_size; i++) {
      if (broadcast_flags1[flag_size - 1 - i]) {
        // broadcast indices are added from left to right
        broadcast_indice1->push_back(indice[i]);
      } else if (flag_size - i <= tensor_a->shape.size() + axis_offset &&
                 broadcast_indice1->size() < tensor_a->shape.size()) {
        broadcast_indice1->push_back(Expr(0));
      }
      if (broadcast_flags2[flag_size - 1 - i]) {
        broadcast_indice2->push_back(indice[i]);
      } else if (flag_size - i <= tensor_b->shape.size() + axis_offset &&
                 broadcast_indice2->size() < tensor_b->shape.size()) {
        // insert indice 0 when have not yet reached the dimension of tensor. Meanwhile we have to consider the case of
        // axis alignment.
        broadcast_indice2->push_back(Expr(0));
      }
    }
  }
}

template <typename FuncOp>
Tensor Broadcast(const FuncOp& op,
                 const Tensor& a,
                 const Tensor& b,
                 const std::string& output_name = "",
                 const Expr& axis               = Expr(-1)) {
  std::vector<Expr> common_shape;
  std::vector<bool> broadcast_flags1;
  std::vector<bool> broadcast_flags2;

  // the counts of left-shift of tensor b so as to right alignment
  int axis_offset = 0;

  GetBroadcastShape(a->shape, b->shape, &common_shape, &broadcast_flags1, &broadcast_flags2, &axis_offset, axis);
  auto fn = [=](const std::vector<Expr>& indice) {
    std::vector<Expr> broadcast_indice1;
    std::vector<Expr> broadcast_indice2;
    GetBroadcastIndice(
        indice, a, b, axis_offset, &broadcast_indice1, &broadcast_indice2, broadcast_flags1, broadcast_flags2);
    return op(a(broadcast_indice1), b(broadcast_indice2));
  };
  Tensor output = Compute(common_shape, fn, output_name);
  return output;
}

#define HLIR_IMP_BC_PE(name__, compute__)                                                             \
  Tensor name__(const Tensor& A, const Tensor& B, const std::string& output_name, const Expr& axis) { \
    auto fn = [&](const Expr& a, const Expr& b) { compute__ };                                        \
    return Broadcast(fn, A, B, output_name, axis);                                                    \
  }

HLIR_IMP_BC_PE(Add, return a + b;);
HLIR_IMP_BC_PE(Substract, return a - b;);
HLIR_IMP_BC_PE(Multiply, return a * b;);
HLIR_IMP_BC_PE(Divide, return a / b;);
HLIR_IMP_BC_PE(FloorDivide, return lang::FloorDivide(a, b););
HLIR_IMP_BC_PE(Remainder, return a.type().is_int() ? a % b : lang::Remainder(a, b););
HLIR_IMP_BC_PE(Mod, return lang::Mod(a, b););
HLIR_IMP_BC_PE(Maximum, return ir::Max::Make(a, b););
HLIR_IMP_BC_PE(Minimum, return ir::Min::Make(a, b););
HLIR_IMP_BC_PE(LeftShift, return a << b;);
HLIR_IMP_BC_PE(RightShift, return a >> b;);
HLIR_IMP_BC_PE(LogicalAnd, return a && b;);
HLIR_IMP_BC_PE(LogicalOr, return a || b;);
HLIR_IMP_BC_PE(LogicalXOr, return (a || b) && !(a && b););
HLIR_IMP_BC_PE(BitwiseAnd, return a & b;);
HLIR_IMP_BC_PE(BitwiseOr, return a | b;);
HLIR_IMP_BC_PE(BitwiseXor, return a ^ b;);
HLIR_IMP_BC_PE(Greater, return a > b;);
HLIR_IMP_BC_PE(Less, return a < b;);
HLIR_IMP_BC_PE(Equal, return ir::EQ::Make(a, b););
HLIR_IMP_BC_PE(NotEqual, return ir::NE::Make(a, b););
HLIR_IMP_BC_PE(GreaterEqual, return a >= b;);
HLIR_IMP_BC_PE(LessEqual, return a <= b;);

Tensor Pow(
    const Tensor& A, const Tensor& B, const std::string& output_name, const Expr& axis, const common::Target& target) {
  CHECK_EQ(A->type(), B->type()) << "The data type of input tensors of pow op should be equal, but here x:" << A->type()
                                 << " != y:" << B->type() << "! Please check.";

  std::string extern_func_name = "cinn_";
  if (target == common::DefaultNVGPUTarget()) {
    extern_func_name += "nvgpu_";
  } else if (target == common::DefaultHostTarget()) {
    extern_func_name += "host_";
  } else {
    LOG(FATAL) << "Pow op only support nvgpu and host now, but here " << target << "! Please check.";
  }

  extern_func_name += "pow_";

  if (A->type().is_float(64)) {
    extern_func_name += "fp64";
  } else if (A->type().is_float(32)) {
    extern_func_name += "fp32";
  } else if (A->type().is_int(32)) {
    extern_func_name += "int32";
  } else {
    LOG(FATAL) << "Pow op only support float64/float32/int32 now, but here " << A->type() << "! Please check.";
  }

  auto fn = [&](const Expr& a, const Expr& b) { return lang::CallExtern(extern_func_name, {a, b}); };
  return Broadcast(fn, A, B, output_name, axis);
}

Tensor BroadcastTo(const Tensor& A,
                   const std::vector<int>& out_shape,
                   const std::vector<int>& broadcast_axes,
                   const std::string& out_name) {
  auto A_shape = A->shape;
  CHECK_EQ(A_shape.size(), broadcast_axes.size()) << "broadcast_axes's size should be same with the input shape's size";
  CHECK_GE(out_shape.size(), broadcast_axes.size()) << "broadcast_axes's size should be no more than out_shape's size";

  return Compute(
      ToCinnExprs(out_shape),
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> broadcast_indice;
        for (int i = 0; i < broadcast_axes.size(); i++) {
          int a_shape_i = A_shape[i].as_int32();
          CHECK(broadcast_axes[i] >= 0 && broadcast_axes[i] < out_shape.size())
              << "broadcast_axis should be no less than 0 and no more than out_shape's dim. Current broadcast axis is "
              << broadcast_axes[i];
          CHECK(a_shape_i == 1 || a_shape_i == out_shape[broadcast_axes[i]])
              << "broadcast_shape should be 1 or same with the target mapping dim, but get " << A_shape[i] << " and "
              << out_shape[broadcast_axes[i]];
          broadcast_indice.push_back(indice[broadcast_axes[i]] % A_shape[i]);
        }
        return A(broadcast_indice);
      },
      out_name);
}

ir::Tensor IsClose(
    const ir::Tensor& x, const ir::Tensor& y, float rtol, float atol, bool equal_nan, const std::string& out_name) {
  CHECK_EQ(x->shape.size(), y->shape.size())
      << "The two inputs shape dimension of is close should be equal! Please check.";

  std::vector<int> x_shape, y_shape;
  for (int i = 0; i < x->shape.size(); ++i) {
    x_shape.emplace_back(x->shape[i].as_int32());
    y_shape.emplace_back(y->shape[i].as_int32());
  }

  CHECK(x_shape == y_shape) << "The two inputs shape of isclose should be equal, but here x:["
                            << cinn::utils::Join(x_shape, ",") << "] != y:[" << cinn::utils::Join(y_shape, ",")
                            << "] ! Please check.";

  // For each a=x[i], b=y[i]:
  // ```
  // if (isnan(a) || isnan(b)) {
  //   out = equal_nan && isnan(a) == isnan(b);
  // } else {
  //   T left = (a > b ? a - b : b - a);
  //   T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
  //   T diff = (left > right ? left - right : right - left);
  //   out = a == b || left <= right || diff <= 1e-15;
  // }
  // ```
  return Compute(
      x->shape,
      [=](const std::vector<Expr>& indice) {
        // check whether x or y is nan
        auto a = x(indice), b = y(indice);
        auto check_x_nan = lang::IsNan(a);
        auto check_y_nan = lang::IsNan(b);

        // out = equal_nan && isnan(a) == isnan(b);
        auto check_nan_same = Expr(equal_nan) && ir::EQ::Make(check_x_nan, check_y_nan);

        // check whether x and y are close
        // T left = (a > b ? a - b : b - a);
        auto left = ir::Select::Make(a > b, a - b, b - a);
        // T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
        auto right = atol + ir::Select::Make(b > 0.0f, rtol * b, (-rtol) * b);
        // T diff = (left > right ? left - right : right - left);
        auto diff = ir::Select::Make(left > right, left - right, right - left);
        // out = a == b || left <= right || diff <= 1e-15;
        auto check_diff = (ir::EQ::Make(a, b) || (left <= right)) || (diff <= 1e-15f);

        return ir::Select::Make(check_x_nan || check_y_nan, check_nan_same, check_diff);
      },
      out_name);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

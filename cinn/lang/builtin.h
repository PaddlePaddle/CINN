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

#include <vector>

#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace lang {

//! Get the ALL of the conditions.
Expr logic_and(const std::vector<Expr>& conds);
Expr logic_or(const std::vector<Expr>& conds);

//! extern call op
#define EXTERN_CALL_DCL(name__) Expr name__(Expr e);

EXTERN_CALL_DCL(Exp);
EXTERN_CALL_DCL(Erf);
EXTERN_CALL_DCL(Sqrt);
EXTERN_CALL_DCL(Rsqrt);
EXTERN_CALL_DCL(Log);
EXTERN_CALL_DCL(Log2);
EXTERN_CALL_DCL(Log10);
EXTERN_CALL_DCL(Floor);
EXTERN_CALL_DCL(Ceil);
EXTERN_CALL_DCL(Round);
EXTERN_CALL_DCL(Trunc);
EXTERN_CALL_DCL(Cos);
EXTERN_CALL_DCL(Cosh);
EXTERN_CALL_DCL(Tan);
EXTERN_CALL_DCL(Sin);
EXTERN_CALL_DCL(Sinh);
EXTERN_CALL_DCL(Acos);
EXTERN_CALL_DCL(Acosh);
EXTERN_CALL_DCL(Asin);
EXTERN_CALL_DCL(Asinh);
EXTERN_CALL_DCL(Atan);
EXTERN_CALL_DCL(Atanh);
EXTERN_CALL_DCL(Tanh);

inline Expr Sigmoid(Expr e) {
  auto one = common::make_const(e->type(), 1);
  return one / (one + Exp(-e));
}

inline Expr Sign(Expr e) {
  auto zero    = make_const(e->type(), 0);
  auto one     = make_const(e->type(), 1);
  auto neg_one = make_const(e->type(), -1);
  auto ret1    = ir::Select::Make(e > zero, one, zero);
  auto ret2    = ir::Select::Make(e < zero, neg_one, ret1);
  return ret2;
}

Expr Abs(Expr e);

inline Expr Negative(Expr e) { return -e; }
inline Expr Identity(Expr e) { return e; }
inline Expr LogicalNot(Expr e) { return !e; }
inline Expr BitwiseNot(Expr e) { return ~e; }
inline Expr BitwiseAnd(Expr a, Expr b) { return a & b; }
inline Expr BitwiseOr(Expr a, Expr b) { return a | b; }
inline Expr BitwiseXor(Expr a, Expr b) { return a ^ b; }
inline Expr LeftShift(Expr a, Expr b) { return a << b; }
inline Expr RightShift(Expr a, Expr b) { return a >> b; }

template <typename T>
inline Expr Relu(Expr e, T threshold = static_cast<T>(0)) {
  return ir::Max::Make(e, make_const(e->type(), threshold));
}

template <typename T>
inline Expr Relu6(Expr e, T threshold = static_cast<T>(0)) {
  return ir::Min::Make(ir::Max::Make(e, make_const(e->type(), threshold)), make_const(e->type(), 6));
}

inline Expr LeakyRelu(Expr e, double alpha) {
  auto zero = make_const(e->type(), 0);
  return ir::Select::Make(e > zero, e, e * make_const(e->type(), alpha));
}

inline Expr LeakyRelu(Expr e, Expr alpha) {
  auto zero = make_const(e->type(), 0);
  return ir::Select::Make(e > zero, e, e * alpha);
}

inline Expr ReduceSum(Expr e, const std::vector<Var>& reduce_axis, Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = make_const(e->type(), 0.f);
  }
  return ir::Reduce::Make(ir::Reduce::kSum, initial, e, reduce_axis);
}

inline Expr ReduceMul(Expr e, const std::vector<Var>& reduce_axis, Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = make_const(e->type(), 1);
  }
  return ir::Reduce::Make(ir::Reduce::kMul, initial, e, reduce_axis);
}

Expr min_value(const Type& type);
Expr max_value(const Type& type);

inline Expr ReduceMax(Expr e, const std::vector<Var>& reduce_axis, Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = min_value(e.type());
  }
  return ir::Reduce::Make(ir::Reduce::kMax, initial, e, reduce_axis);
}
inline Expr ReduceMin(Expr e, const std::vector<Var>& reduce_axis, Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = max_value(e.type());
  }
  return ir::Reduce::Make(ir::Reduce::kMin, initial, e, reduce_axis);
}

Expr IsNan(Expr e);

Expr Infinity(const Type& type);

Expr IsInf(Expr e);

Expr IsFinite(Expr e);

}  // namespace lang
}  // namespace cinn

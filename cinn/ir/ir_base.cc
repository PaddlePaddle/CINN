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

#include "cinn/ir/ir_base.h"

#include "cinn/common/cinn_value.h"
#include "cinn/common/common.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/module.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace ir {

//! Implementations for Ir Expr Nodes.
// @{
#define __m(t__)                                             \
  template <>                                                \
  void ExprNode<t__>::Accept(cinn::ir::IRVisitor *v) const { \
    v->Visit(const_self());                                  \
  }
#undef __m
// @}

std::ostream &operator<<(std::ostream &os, IrNodeTy type) {
  switch (type) {
#define __m(t__)                    \
  case IrNodeTy::t__:               \
    os << "<node: " << #t__ << ">"; \
    break;

    NODETY_FORALL(__m)
#undef __m

    default:
      LOG(FATAL) << "unknown IrNodeTy found";
  }

  return os;
}

Expr::Expr(const Var &var) { *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&var); }

int32_t Expr::as_int32() const {
  CHECK(type().is_int(32));
  return As<IntImm>()->value;
}
int64_t Expr::as_int64() const {
  CHECK(type().is_int(64));
  return As<IntImm>()->value;
}
float Expr::as_float() const {
  CHECK(type().is_float(32));
  return As<FloatImm>()->value;
}
double Expr::as_double() const {
  CHECK(type().is_float(64));
  return As<FloatImm>()->value;
}

Expr &Expr::operator=(const Expr &other) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

Expr::operator Var() {
  auto *x = As<ir::_Var_>();
  CHECK(x);
  return ir::Var(x);
}

bool Expr::is_constant() const { return As<IntImm>() || As<UIntImm>() || As<FloatImm>(); }

double Expr::get_constant() const {
  CHECK(is_constant()) << *this << " is not constant! Please check.";
  auto *vi = As<IntImm>();
  auto *vf = As<FloatImm>();
  if (vi) return vi->value;
  return vf->value;
}

bool Expr::is_var() const { return As<_Var_>(); }

_Buffer_ *Expr::as_buffer() { return As<_Buffer_>(); }
const _Buffer_ *Expr::as_buffer() const { return As<_Buffer_>(); }
Buffer Expr::as_buffer_ref() const { return Buffer(&Reference(as_buffer())); }

_LoweredFunc_ *Expr::as_lowered_func() { return As<_LoweredFunc_>(); }
const _LoweredFunc_ *Expr::as_lowered_func() const { return As<_LoweredFunc_>(); }

_Module_ *Expr::as_module() { return As<_Module_>(); }
const _Module_ *Expr::as_module() const { return As<_Module_>(); }
ir::Module Expr::as_module_ref() const {
  auto *module = as_module();
  CHECK(module);  // Need check here?
  // TODO(Superjomn) remove the Reference here.
  return ir::Module(&Reference(module));
}

LoweredFunc Expr::as_lowered_func_ref() const {
  auto *function = as_lowered_func();
  CHECK(function);
  return LoweredFunc(&Reference(function));
}

_Tensor_ *Expr::as_tensor() { return As<_Tensor_>(); }
const _Tensor_ *Expr::as_tensor() const { return As<_Tensor_>(); }
ir::Tensor Expr::as_tensor_ref() const { return ir::Tensor(&Reference(as_tensor())); }

_Var_ *Expr::as_var() { return As<_Var_>(); }
const _Var_ *Expr::as_var() const { return As<_Var_>(); }
Var Expr::as_var_ref() const { return Var(&Reference(as_var())); }

bool Expr::is_cmp() const {
  switch (node_type()) {
    case ir::IrNodeTy::LE:
    case ir::IrNodeTy::LT:
    case ir::IrNodeTy::EQ:
    case ir::IrNodeTy::NE:
    case ir::IrNodeTy::GT:
    case ir::IrNodeTy::GE:
      return true;
    default:
      return false;
  }
}

const Expr &IrNode::operand(int i) {
  CHECK_LT(i, operands.size());
  return operands[i];
}

}  // namespace ir
}  // namespace cinn

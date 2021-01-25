#pragma once

#include <glog/logging.h>
#include <llvm/IR/Intrinsics.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/ir/intrinsic_ops.h"
#include "cinn/ir/registry.h"
#include "cinn/lang/packed_func.h"

namespace cinn {
namespace codegen {

template <int id, int arg_nums, bool add_float_suffix = true>
inline void MakeFloatIntrinOp(lang::Args args, lang::RetValue *rv) {
  CHECK_GE(args.size(), 1U);
  Expr arg       = args[0];
  ir::Call *node = arg->as<ir::Call>();
  CHECK(node);
  CHECK_GE(node->read_args.size(), arg_nums);
  if (add_float_suffix) {
    CHECK(node->type().is_float());
    *rv = ir::intrinsics::BuiltinIntrin::Make(node->name + "f", node->read_args, id, arg_nums, node->type());
  } else {
    *rv = ir::intrinsics::BuiltinIntrin::Make(node->name, node->read_args, id, arg_nums, node->type());
  }
}

void RegisterCpuIntrinRule() {
#define __(intrin_name__, id) \
  ir::Registry::Register("lower_cpu_intrinsic_" #intrin_name__, true).SetBody(MakeFloatIntrinOp<id, 1>);
  __(exp, ::llvm::Intrinsic::exp)
  __(exp2, ::llvm::Intrinsic::exp2)
  __(sqrt, ::llvm::Intrinsic::sqrt)
  __(log, ::llvm::Intrinsic::log)
  __(log2, ::llvm::Intrinsic::log2)
  __(log10, ::llvm::Intrinsic::log10)
  __(floor, ::llvm::Intrinsic::floor)
  __(ceil, ::llvm::Intrinsic::ceil)
  __(round, ::llvm::Intrinsic::round)
  __(trunc, ::llvm::Intrinsic::trunc)
  __(cos, ::llvm::Intrinsic::cos)
  __(sin, ::llvm::Intrinsic::sin)
  __(fabs, ::llvm::Intrinsic::fabs)
#undef __

// set id -1 if not llvm intrinsics
#define RegisterBitwise(intrin_name__) \
  ir::Registry::Register("lower_cpu_intrinsic_" #intrin_name__, true).SetBody(MakeFloatIntrinOp<-1, 2, false>);
  RegisterBitwise(bitwise_or) RegisterBitwise(bitwise_xor) RegisterBitwise(bitwise_and) RegisterBitwise(left_shift)
      RegisterBitwise(right_shift)
#undef RegisterBitwise

          ir::Registry::Register("lower_cpu_intrinsic_fma", true)
              .SetBody(MakeFloatIntrinOp<::llvm::Intrinsic::fmuladd, 3, false>);

  ir::Registry::Register("lower_cpu_intrinsic_bitwise_not", true).SetBody(MakeFloatIntrinOp<-1, 1, false>);

  ir::Registry::Register("lower_cpu_intrinsic_isnan", true).SetBody(MakeFloatIntrinOp<-1, 1, false>);

  ir::Registry::Register("lower_cpu_intrinsic_isfinite", true).SetBody([](lang::Args args, lang::RetValue *rv) {
    CHECK_GE(args.size(), 1U);
    Expr arg0      = args[0];
    ir::Call *node = arg0->as<ir::Call>();
    CHECK(node);
    CHECK(!node->read_args.empty());
    Expr arg = node->read_args[0];
    *rv      = !(lang::IsInf(arg)) && !(lang::IsNan(arg));
  });

  ir::Registry::Register("lower_cpu_intrinsic_isinf", true).SetBody([](lang::Args args, lang::RetValue *rv) {
    CHECK_GE(args.size(), 1U);
    Expr arg0      = args[0];
    ir::Call *node = arg0->as<ir::Call>();
    CHECK(node);
    CHECK(!node->read_args.empty());
    Expr arg  = node->read_args[0];
    Type type = arg->type();
    if (type.is_int() || type.is_uint()) {
      *rv = common::make_bool(false, type.lanes());
    } else if (type.is_float()) {
      *rv = ir::EQ::Make(lang::Abs(arg), lang::Infinity(type)) && !(lang::IsNan(arg));
    }
  });

  ir::Registry::Register("lower_cpu_intrinsic_exp10", true).SetBody([](lang::Args args, lang::RetValue *rv) {
    CHECK_GE(args.size(), 1U);
    Expr arg0      = args[0];
    ir::Call *node = arg0->as<ir::Call>();
    CHECK(node);
    CHECK(!node->read_args.empty());
    Expr arg  = node->read_args[0];
    Expr ln10 = make_const(arg->type(), 2.302585093);
    *rv       = lang::Exp(arg * ln10);
  });

  ir::Registry::Register("lower_cpu_intrinsic_tan", true).SetBody([](lang::Args args, lang::RetValue *rv) {
    CHECK_GE(args.size(), 1U);
    Expr arg0      = args[0];
    ir::Call *node = arg0->as<ir::Call>();
    CHECK(node);
    CHECK(!node->read_args.empty());
    Expr arg = node->read_args[0];
    *rv      = lang::Sin(arg) / lang::Cos(arg);
  });

  ir::Registry::Register("lower_cpu_intrinsic_tanh", true).SetBody([](lang::Args args, lang::RetValue *rv) {
    CHECK_GE(args.size(), 1U);
    Expr arg0      = args[0];
    ir::Call *node = arg0->as<ir::Call>();
    CHECK(node);
    CHECK(!node->read_args.empty());
    Expr arg     = node->read_args[0];
    Expr zero    = make_const(arg->type(), 0);
    Expr one     = make_const(arg->type(), 1);
    Expr two     = make_const(arg->type(), 2);
    Expr neg_two = make_const(arg->type(), -2);

    Expr exp_neg2x = lang::Exp(neg_two * arg);
    Expr exp_pos2x = lang::Exp(two * arg);

    Expr tanh_pos = (one - exp_neg2x) / (one + exp_neg2x);
    Expr tanh_neg = (exp_pos2x - one) / (exp_pos2x + one);
    *rv           = ir::Select::Make(arg >= zero, tanh_pos, tanh_neg);
  });

  ir::Registry::Register("lower_cpu_intrinsic_cosh", true).SetBody([](lang::Args args, lang::RetValue *rv) {
    CHECK_GE(args.size(), 1U);
    Expr arg0      = args[0];
    ir::Call *node = arg0->as<ir::Call>();
    CHECK(node);
    CHECK(!node->read_args.empty());
    Expr arg = node->read_args[0];
    *rv      = (lang::Exp(arg) + lang::Exp(arg * make_const(arg->type(), -1))) / make_const(arg->type(), 2);
  });

  ir::Registry::Register("lower_cpu_intrinsic_sinh", true).SetBody([](lang::Args args, lang::RetValue *rv) {
    CHECK_GE(args.size(), 1U);
    Expr arg0      = args[0];
    ir::Call *node = arg0->as<ir::Call>();
    CHECK(node);
    CHECK(!node->read_args.empty());
    Expr arg = node->read_args[0];
    *rv      = (lang::Exp(arg) - lang::Exp(arg * make_const(arg->type(), -1))) / make_const(arg->type(), 2);
  });
}
}  // namespace codegen
}  // namespace cinn

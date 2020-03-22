#include "cinn/optim/ir_simplify.h"

#include <ginac/ginac.h>
#include <glog/logging.h>

#include <map>
#include <string>

#include "cinn/common/arithmatic.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using common::ExprToGinacConerter;
using utils::GetStreamCnt;
using utils::Replace;

namespace {

//! Simplify some sub-expression in the `expr`. Due to the simplify strategy just fit several kinds of IR noedes, we
//! partition the original expression to several sub-expression those supported by simplify, and process each of them.
void PartialSimplify(Expr* expr) {
  ExprToGinacConerter converter;
  auto ex = converter(*expr);
  VLOG(4) << "get ex:" << ex;
  *expr = converter.GinacToExpr(ex);
  VLOG(4) << "ex to expr: " << *expr;
}

//! Simplify the expression but Load.
struct SimplifyButStoreLoadMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

#define __(op__)                                    \
  void Visit(const op__* op, Expr* expr) override { \
    auto* node   = expr->As<op__>();                \
    auto* a_ramp = node->a().As<ir::Ramp>();        \
    auto* b_ramp = node->b().As<ir::Ramp>();        \
    if (a_ramp) {                                   \
      PartialSimplify(&a_ramp->base);               \
      PartialSimplify(&a_ramp->stride);             \
    }                                               \
    if (b_ramp) {                                   \
      PartialSimplify(&b_ramp->base);               \
      PartialSimplify(&b_ramp->stride);             \
    }                                               \
    if (!(a_ramp || b_ramp)) {                      \
      PartialSimplify(expr);                        \
    } else if (!a_ramp) {                           \
      PartialSimplify(&node->a());                  \
    } else if (!b_ramp) {                           \
      PartialSimplify(&node->b());                  \
    }                                               \
  }

  __(Add)
  __(Mul)
  __(Sub)
  __(Div)
#undef __

  void Visit(const Ramp* op, Expr* expr) override {
    auto* node = expr->As<Ramp>();
    CHECK(common::IsPureMath(node->base));
    CHECK(common::IsPureMath(node->stride));
    PartialSimplify(&node->base);
    PartialSimplify(&node->stride);
  }
};  // namespace

struct SimplifyLoadMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Load* expr, Expr* op) override {
    auto* node = op->As<Load>();
    if (common::IsPureMath(node->index))
      PartialSimplify(&node->index);
    else {
      SimplifyButStoreLoadMutator mutator;
      mutator(&node->index);
    }
  }
};

struct SimplifyStoreMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Store* expr, Expr* op) override {
    auto* node = op->As<Store>();

    if (common::IsPureMath(node->index)) {
      PartialSimplify(&node->index);
    } else {
      SimplifyButStoreLoadMutator mutator;
      mutator(&node->index);
    }
  }
};

}  // namespace

void Simplify(Expr* expr) {
  SimplifyLoadMutator()(expr);
  SimplifyStoreMutator()(expr);
  SimplifyButStoreLoadMutator()(expr);
}

}  // namespace optim
}  // namespace cinn

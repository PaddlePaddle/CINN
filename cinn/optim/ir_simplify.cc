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

struct SimplifyStoreMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Store* expr, Expr* op) override {
    auto* node = op->As<Store>();
    VLOG(4) << "to simplify Load: " << *op;
    PartialSimplify(&node->index);
    VLOG(4) << "get: " << *op;
  }
};

struct SimplifyLoadMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Load* expr, Expr* op) override {
    auto* node = op->As<Load>();
    VLOG(4) << "to simplify Load: " << *op;
    PartialSimplify(&node->index);
    VLOG(4) << "get: " << *op;
  }
};

//! Simplify the expression but Load.
struct SimplifyButStoreLoadMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Add* expr, Expr* op) override {
    VLOG(4) << "origin: " << *op;
    PartialSimplify(op);
    VLOG(4) << "simplified: " << *op;
  }
  void Visit(const Sub* expr, Expr* op) override {
    VLOG(4) << "origin: " << *op;
    PartialSimplify(op);
    VLOG(4) << "simplified: " << *op;
  }
  void Visit(const Mul* expr, Expr* op) override {
    VLOG(4) << "origin: " << *op;
    PartialSimplify(op);
    VLOG(4) << "simplified: " << *op;
  }
  void Visit(const Div* expr, Expr* op) override {
    VLOG(4) << "origin: " << *op;
    PartialSimplify(op);
    VLOG(4) << "simplified: " << *op;
  }

#undef __
};

}  // namespace

void Simplify(Expr* expr) {
  SimplifyLoadMutator()(expr);
  SimplifyStoreMutator()(expr);
  SimplifyButStoreLoadMutator()(expr);
}

}  // namespace optim
}  // namespace cinn

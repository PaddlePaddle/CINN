#include "cinn/optim/compare_simplify.h"

#include <string>

#include "cinn/common/ir_util.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

namespace cinn::optim {

namespace {

struct IslPrinter : public ir::IrPrinter {
  using ir::IrPrinter::Print;

  explicit IslPrinter(std::ostream& os) : ir::IrPrinter(os) {}

  void Visit(const ir::EQ* op) {
    Print(op->a());
    os() << " = ";
    Print(op->b());
  }

  void Visit(const ir::Mod* op) {
    Print(op->a());
    os() << " mod ";
    Print(op->b());
  }
};

/**
 * Static immediate constant condition(no variable is involved) simplify.
 * e.g.
 * (1 > 2) will be simplified to false
 * (1 < 2) will be simplified to true
 */
struct StaticImmConditionMutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::LE* op, Expr* expr) {
    if (!Simplify(expr)) ir::IRMutator<>::Visit(op, expr);
  }
  void Visit(const ir::LT* op, Expr* expr) {
    if (!Simplify(expr)) ir::IRMutator<>::Visit(op, expr);
  }
  void Visit(const ir::GE* op, Expr* expr) {
    if (!Simplify(expr)) ir::IRMutator<>::Visit(op, expr);
  }
  void Visit(const ir::GT* op, Expr* expr) {
    if (!Simplify(expr)) ir::IRMutator<>::Visit(op, expr);
  }
  void Visit(const ir::EQ* op, Expr* expr) {
    if (!Simplify(expr)) ir::IRMutator<>::Visit(op, expr);
  }
  void Visit(const ir::NE* op, Expr* expr) {
    if (!Simplify(expr)) ir::IRMutator<>::Visit(op, expr);
  }
  void Visit(const ir::Mod* op, Expr* expr) {
    if (auto* ai = op->a().As<ir::IntImm>()) {
      if (auto* bi = op->b().As<ir::IntImm>()) {
        *expr = Expr(new ir::IntImm(op->a().type(), ai->value % bi->value));
      } else {
        ir::IRMutator<>::Visit(op, expr);
      }
    } else {
      ir::IRMutator<>::Visit(op, expr);
    }
  }

  bool Simplify(Expr* expr) {
    // Can't deal with the condition with variables.
    auto has_var = !ir::CollectIRNodes(*expr, [](const Expr* x) { return x->as_var(); }).empty();
    if (has_var) return false;

    switch (expr->node_type()) {
#define __GET_FIELDS(ntype__, cmp__)                                                 \
  auto* node = expr->As<ir::ntype__>();                                              \
  auto* ai   = node->a().As<ir::IntImm>();                                           \
  auto* bi   = node->b().As<ir::IntImm>();                                           \
  auto* af   = node->a().As<ir::FloatImm>();                                         \
  auto* bf   = node->b().As<ir::FloatImm>();                                         \
  if (ai && bi) {                                                                    \
    *expr = common::make_bool(ai->value cmp__ bi->value, node->a()->type().lanes()); \
    return true;                                                                     \
  }                                                                                  \
  if (af && bf) {                                                                    \
    *expr = common::make_bool(af->value cmp__ bf->value, node->a()->type().lanes()); \
    return true;                                                                     \
  }                                                                                  \
  break;

      case ir::IrNodeTy::EQ: {
        __GET_FIELDS(EQ, ==)
      }
      case ir::IrNodeTy::LT: {
        __GET_FIELDS(LT, <)
      }
      case ir::IrNodeTy::LE: {
        __GET_FIELDS(LE, <=)
      }
      case ir::IrNodeTy::GT: {
        __GET_FIELDS(GT, >)
      }
      case ir::IrNodeTy::GE: {
        __GET_FIELDS(GE, >=)
      }
      case ir::IrNodeTy::NE: {
        __GET_FIELDS(NE, !=)
      }
      default:
        break;
    }

    return false;
  }
};

}  // namespace

void CompareSimplify(Expr* e) {
  StaticImmConditionMutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim

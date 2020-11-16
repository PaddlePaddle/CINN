#include "cinn/optim/compare_simplify.h"
#include <string>
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

namespace cinn::optim {

namespace {

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

  bool Simplify(Expr* expr) {
    // Can't deal with the condition with variables.
    auto has_var = !ir::CollectIRNodes(*expr, [](const Expr* x) { return x->as_var(); }).empty();
    if (has_var) return false;

    // create a set

    std::string set_repr = utils::StringFormat("[]->{ : %s }", utils::GetStreamCnt(*expr).c_str());
    LOG(INFO) << set_repr;
    isl::ctx ctx(isl_ctx_alloc());
    isl::set cond_set(ctx, set_repr);
    LOG(INFO) << "cond_set: " << cond_set;

    *expr = Expr(!cond_set.is_empty());
    return true;
  }
};

}  // namespace

void CompareSimplify(Expr* e) {
  StaticImmConditionMutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim

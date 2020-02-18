#include "cinn/optim/replace_var_with_expr.h"

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace optim {

struct ReplaceVarWithExprMutator : public ir::IRMutator {
  ReplaceVarWithExprMutator(const Var& var, const Expr& expr) : var_(var), expr_(expr) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (expr->name != var_->name) return;
    auto copied = IRCopy(expr_);
    *op         = copied;
  }

 private:
  const Var& var_;
  const Expr& expr_;
};

void ReplaceVarWithExpr(Expr* source, const Var& var, const Expr& expr) {
  ReplaceVarWithExprMutator mutator(var, expr);
  mutator(source);
}

}  // namespace optim
}  // namespace cinn

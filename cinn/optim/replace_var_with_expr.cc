#include "cinn/optim/replace_var_with_expr.h"

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace optim {

struct ReplaceVarWithExprMutator : public ir::IRMutator<> {
  ReplaceVarWithExprMutator(const Var& var, const Expr& expr) : var_(var), expr_(expr) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (expr->name != var_->name) return;
    auto copied = IRCopy(expr_);
    *op         = copied;
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->min, &node->min);
    ir::IRMutator<>::Visit(&node->extent, &node->extent);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->loop_var->name == var_->name && expr_.As<ir::_Var_>()) {
      node->loop_var = expr_.As<ir::_Var_>();
    }
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    ir::IRMutator<>::Visit(&node->init, &node->init);
    ir::IRMutator<>::Visit(&node->condition, &node->condition);
    ir::IRMutator<>::Visit(&node->inc, &node->inc);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->iterator->name == var_->name && expr_.As<ir::_Var_>()) {
      node->iterator = expr_.As<ir::_Var_>();
    }
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

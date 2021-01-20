#include "cinn/optim/simplify_identity_domain_forloop.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

void SimplifyIdentityDomainForloop(Expr *e) {
  struct Mutator : public ir::IRMutator<> {
    std::map<Var, Expr> iter_to_val;

    using ir::IRMutator<>::Visit;

    void Visit(const ir::For *op, Expr *expr) override {
      auto *node = expr->As<ir::For>();

      if (op->extent.is_constant() && op->extent.as_int32() == 1) {  // to simplify
        iter_to_val[op->loop_var] = op->min;
        ir::IRMutator<>::Visit(&node->body, &node->body);

        // remove outer forloop
        *expr = node->body;

        iter_to_val.erase(op->loop_var);
      } else {
        ir::IRMutator<>::Visit(&node->body, &node->body);
      }
    }

    void Visit(const ir::Store *op, Expr *expr) override {
      auto *node = expr->As<ir::Store>();
      for (auto &e : node->indices) {
        for (auto &[var, expr] : iter_to_val) {
          ReplaceVarWithExpr(&e, var, expr);
        }
      }
    }

    void Visit(const ir::Load *op, Expr *expr) override {}
  };

  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace optim
}  // namespace cinn
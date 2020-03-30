#include "cinn/optim/unroll_loops.h"

#include <vector>

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_replace.h"

namespace cinn {
namespace optim {

namespace {

struct UnrollMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::For* op, Expr* expr) override {
    if (is_unrollable(op)) {
      Unroll(op, expr);
      IRMutator<>::Visit(expr, expr);
    } else {
      auto* node = expr->As<ir::For>();
      ir::IRMutator<>::Visit(&node->body, &node->body);
    }
  }

  bool is_unrollable(const ir::For* op) const {
    return op->is_unrolled() && op->extent.is_constant() && op->extent.as_int32() < 50;
  }

  //! Unroll a forloop.
  void Unroll(const ir::For* op, Expr* expr) {
    std::vector<Expr> body;

    auto* min    = op->min.As<ir::IntImm>();
    auto* extent = op->extent.As<ir::IntImm>();
    if (!(min && extent)) return;

    for (int i = min->value; i < extent->value; i++) {
      Expr start = op->min + i;
      body.push_back(optim::IRCopy(op->body));
      optim::IrReplace(&body.back(), op->loop_var, start);
    }

    *expr = ir::Block::Make(body);
  }
};

}  // namespace

void UnrollLoop(Expr* expr) { UnrollMutator()(expr); }

}  // namespace optim
}  // namespace cinn

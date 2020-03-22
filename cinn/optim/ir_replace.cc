#include "cinn/optim/ir_replace.h"

#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

namespace {

struct IrReplaceMutator : ir::IRMutator<Expr*> {
  IrReplaceMutator(const ir::Var& v, const Expr& expr) : var_(v), expr_(expr) {}
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* op, Expr* expr) override {
    if (var_->name == op->name) {
      *expr = expr_;
    }
  }

  ir::Var var_;
  Expr expr_;
};

}  // namespace

void IrReplace(ir::Expr* expr, ir::Var v, ir::Expr e) { IrReplaceMutator(v, e)(expr); }

}  // namespace optim
}  // namespace cinn

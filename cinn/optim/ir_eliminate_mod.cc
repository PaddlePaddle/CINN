#include "cinn/optim/ir_eliminate_mod.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace optim {

void IrEliminateMod(Expr* expr) {
  struct Modifier : public ir::IRMutator<Expr*> {
    void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::Mod* op, Expr* expr) {
      LOG(ERROR) << "Not Implemented";
      LOG(INFO) << "elimiate " << *expr << " to " << op->a();
      *expr = op->a();
    }
  };

  Modifier()(expr);
}

}  // namespace optim
}  // namespace cinn

#include "cinn/optim/activate_to_extern_call.h"

#include "cinn/cinn.h"
#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

void ActivateToExternCall(Expr *e) {
  struct Mutator : ir::IRMutator<Expr *> {
    void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

    void Visit(const ir::Activate *op, Expr *expr) override {
      auto *node = expr->As<ir::Activate>();
      operator()(&node->operand(0));

      switch (node->kind) {
        case ir::Activate::Kind::kTanh:
          *expr = lang::CallExtern("tanh", {node->operand(0)});
          break;
        default:
          NOT_IMPLEMENTED
      }
    }
  };

  Mutator()(e);
}

}  // namespace optim
}  // namespace cinn

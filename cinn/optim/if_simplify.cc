#include "cinn/optim/if_simplify.h"
#include "cinn/ir/ir_mutator.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::IfThenElse* op, Expr* expr) {
    if (auto* i = op->condition.As<ir::IntImm>()) {
      if (i->value) {
        *expr = op->true_case;
      } else {
        if (op->false_case.defined()) {
          *expr = op->false_case;
        } else {
          // null condition
          *expr = ir::Block::Make({});
        }
      }
    }
  }
};

}  // namespace

void IfSimplify(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim

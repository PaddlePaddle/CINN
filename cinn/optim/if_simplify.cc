#include "cinn/optim/if_simplify.h"

#include "cinn/ir/ir_mutator.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::IfThenElse* op, Expr* expr) {
    auto* condition_int  = op->condition.As<ir::IntImm>();
    auto* condition_uint = op->condition.As<ir::UIntImm>();
    int64_t value;
    if (condition_int || condition_uint) {
      if (condition_int) {
        value = condition_int->value;
      } else {
        value = condition_uint->value;
      }
      if (value) {
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

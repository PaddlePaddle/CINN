#include "cinn/optim/cast_bool_to_int8.h"

#include <glog/logging.h>

#include "cinn/ir/ir_mutator.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    CHECK(node);
    auto value = node->value;
    if (op->type().is_bool() && op->value->type().is_bool()) {
      value = ir::Cast::Make(Int(8), value);
      *expr = ir::Store::Make(node->tensor, value, node->indices);
    }
  }
};

}  // namespace

void CastBoolToInt8(Expr* e, Target target) {
  if (target.arch == Target::Arch::X86) {
    Mutator mutator;
    mutator.Visit(e, e);
  }
}
}  // namespace cinn::optim

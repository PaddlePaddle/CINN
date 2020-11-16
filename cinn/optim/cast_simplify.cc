#include "cinn/optim/cast_simplify.h"
#include "cinn/ir/ir_mutator.h"

namespace cinn::optim {

namespace {

struct Mutator : ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::Cast* op, Expr* expr) {
    auto* node = expr->As<ir::Cast>();

    LOG(INFO) << op->type() << " vs " << op->v().type();
    Visit(&node->v(), &node->v());

    if (op->type() == op->v().type()) {
      *expr = op->v();
      return;
    }

#define __CAST_TO_TYPE(type__)                       \
  if (auto* i = op->v().As<ir::IntImm>()) {          \
    *expr = Expr(static_cast<type__>(i->value));     \
  } else if (auto* f = op->v().As<ir::FloatImm>()) { \
    *expr = Expr(static_cast<type__>(f->value));     \
  } else {                                           \
    CINN_NOT_IMPLEMENTED                             \
  }

    if (op->v().is_constant()) {
      if (op->type() == type_of<int32_t>()) {
        __CAST_TO_TYPE(int32_t)
      } else if (op->type() == type_of<int64_t>()) {
        __CAST_TO_TYPE(int64_t)
      } else if (op->type() == type_of<float>()) {
        __CAST_TO_TYPE(float)
      } else if (op->type() == type_of<double>()) {
        __CAST_TO_TYPE(double)
      }
    }
  }
};

}  // namespace

void CastSimplify(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim

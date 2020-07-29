#include "cinn/optim/transform_polyfor_to_for.h"

#include <cmath>
#include <vector>

#include "cinn/common/arithmatic.h"
#include "cinn/common/cas.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"

namespace cinn {
namespace optim {

void PolyForAutoSeparate(Expr* expr);

namespace {

Expr PlusOneWithMinMax(Expr expr) {
  auto* min_n = expr.As<ir::Min>();
  auto* max_n = expr.As<ir::Max>();

  if (min_n) {
    min_n->a() = min_n->a() + 1;
    min_n->b() = min_n->b() + 1;
    Simplify(&min_n->a());
    Simplify(&min_n->b());
    return expr;
  } else if (max_n) {
    max_n->a() = max_n->a() + 1;
    max_n->b() = max_n->b() + 1;
    Simplify(&max_n->a());
    Simplify(&max_n->b());
    return expr;
  }
  return expr + 1;
}

struct PolyForWithSimpleConditionToForMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* lt_n = op->condition.As<ir::LT>();
    auto* le_n = op->condition.As<ir::LE>();

    if (!(lt_n || le_n)) return;

    // check the lhs is the iterator
    bool can_extract_extent = (lt_n && lt_n->a().as_var() && lt_n->a().as_var()->name == op->iterator->name) ||
                              (le_n && le_n->a().as_var() && le_n->a().as_var()->name == op->iterator->name);
    if (can_extract_extent) {
      Expr lhs = lt_n ? lt_n->a() : le_n->a();
      Expr rhs = lt_n ? lt_n->b() : PlusOneWithMinMax(le_n->b());
      rhs      = common::AutoSimplify(rhs);

      if (op->is_vectorized()) CHECK(op->vectorize_info().valid());

      Expr new_for =
          ir::For::Make(op->iterator, op->init, rhs, op->for_type(), op->device_api, op->body, op->vectorize_info());
      *expr = new_for;

      Visit(&new_for.As<ir::For>()->body);
    }
  }
};

}  // namespace

namespace detail {

void PolyForWithSimpleConditionToFor(Expr* expr) {
  PolyForWithSimpleConditionToForMutator mutator;
  mutator(expr);
}

void PolyForAutoSeparate(Expr* expr) {
  ForAutoSeparateMutatorMain main;
  main(expr);
}

}  // namespace detail

void TransformPolyForToFor(Expr* expr, bool auto_separate) {
  detail::PolyForWithSimpleConditionToFor(expr);
  if (auto_separate) detail::PolyForAutoSeparate(expr);
}

}  // namespace optim
}  // namespace cinn

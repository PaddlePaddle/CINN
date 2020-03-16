#include "cinn/optim/transform_polyfor_to_for.h"
#include <stack>
#include <vector>
#include "cinn/common/arithmatic.h"
#include "cinn/common/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/optim/ir_simplify.h"

namespace cinn {
namespace optim {

void PolyForAutoSeparate(Expr* expr);

namespace {

//! NOTE The input expressions can only deal with PolyFor, not For nodes.
struct PolyForAutoSeparateMutator : ir::IRMutator<Expr*> {
  void Visit(const ir::PolyFor* op, Expr* expr) override {
    forloop_stack.push_back(expr);

    Expr condition = op->condition;

    auto* le_n = condition.As<ir::LE>();
    auto* lt_n = condition.As<ir::LT>();

    do {  // We use a do-while here to break in any position and continue the post-procession after the while block.
      if (le_n || lt_n) {
        Expr lhs = le_n ? le_n->a : lt_n->a;
        Expr rhs = le_n ? le_n->b : lt_n->b;

        // assert the lhs is the iterator
        if (lhs != Expr(op->iterator)) break;

        auto* min_n = rhs.As<ir::Min>();
        if (!min_n) break;
        // TODO(Superjomn) We can support max latter.

        Expr left  = min_n->a;
        Expr right = min_n->b;

        CHECK(common::IsPureMath(left));
        CHECK(common::IsPureMath(right));

        // find the forloop level to separate
        std::vector<int> separate_levels;
        int level = 0;
        for (auto& _forloop : forloop_stack) {
          auto* forloop = _forloop->As<ir::PolyFor>();
          bool contains = common::MathContainsSymbol(left, forloop->iterator) ||
                          common::MathContainsSymbol(right, forloop->iterator);
          if (contains) separate_levels.push_back(level);
          level++;
        }
        //! too complex
        if (separate_levels.size() > 1) break;

        auto forloop_to_separate = forloop_stack[separate_levels.front()]->As<ir::PolyFor>();

        // check the min not include the current iterator, or it is illegal.
        Expr solve_res;
        bool is_positive;
        // solve left <= right
        std::tie(solve_res, is_positive) = common::Solve(right, left, forloop_to_separate->iterator);
        // iterator >= solve_res
      }
    } while (false);
  }

  //! Separate the PolyFor into two PolyFors.
  void SeparateForloop(Expr* poly_for_expr, Expr upper_bound) {
    auto* node = poly_for_expr->As<ir::For>();
    CHECK(node);
    CHECK(common::is_zero(node->min));

    Expr body = node;
  }

  //! Stack of the forloops.
  std::vector<Expr*> forloop_stack;
};

struct PolyForWithSimpleConditionToForMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }
  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* lt_n = op->condition.As<ir::LT>();
    auto* le_n = op->condition.As<ir::LE>();

    if (!(lt_n || le_n)) return;

    Expr lhs = lt_n ? lt_n->a : le_n->a;
    Expr rhs = lt_n ? lt_n->b : le_n->b + 1;
    if (common::IsPureMath(rhs)) optim::Simplify(&rhs);

    CHECK(lhs == Expr(op->iterator));
    CHECK(op->inc == Expr(1));

    Expr new_for = ir::For::Make(op->iterator, op->init, rhs, op->for_type, op->device_api, op->body);
    *expr        = new_for;

    Visit(&new_for.As<ir::For>()->body);
  }
};

}  // namespace

namespace detail {

void PolyForWithSimpleConditionToFor(Expr* expr) {
  PolyForWithSimpleConditionToForMutator mutator;
  mutator(expr);
}

}  // namespace detail

}  // namespace optim
}  // namespace cinn

#include "cinn/optim/transform_polyfor_to_for.h"

#include <cmath>
#include <stack>
#include <vector>

#include "cinn/common/arithmatic.h"
#include "cinn/common/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"

namespace cinn {
namespace optim {

void PolyForAutoSeparate(Expr* expr);

namespace {

/**
 * Separate a forloop in the expression.
 *                     *forloop(min, extent) -- to separate
 *                         |
 *                      forloop
 *                         |
 *                     forloop(min, Min(a,b))
 * will be rewriten to
 *                      forloop(min, separator) ---  forloop(separator, extent)
 *                         |                               |
 *                      forloop                         forloop
 *                         |                               |
 *                   forloop(min, a)                  forloop(min, b)
 */
struct ForSeparater : ir::IRMutator<Expr*> {
  //! @param forloop The forloop to separate.
  //! @param separator The separator to split the domain of the \p forloop.
  //! @param sub_forloop The forloop whose extent has a Min node.
  ForSeparater(Expr* forloop, Expr separator, ir::For* sub_forloop)
      : forloop_(forloop), separator_(separator), sub_forloop_(sub_forloop) {}

  void operator()(Expr* expr) { Visit(expr); }

 private:
  void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    if (expr == forloop_) {  // find the forloop to split
      is_separated_ = true;

      auto forloop_branch0 =
          ir::For::Make(op->loop_var, op->min, separator_, op->for_type, op->device_api, optim::IRCopy(op->body));
      is_left_branch_ = true;
      Visit(&forloop_branch0.As<ir::For>()->body);

      auto forloop_branch1 =
          ir::For::Make(op->loop_var, separator_, op->extent, op->for_type, op->device_api, optim::IRCopy(op->body));
      is_left_branch_ = false;
      Visit(&forloop_branch1.As<ir::For>()->body);

      *expr = ir::Block::Make({forloop_branch0, forloop_branch1});
    } else if (MatchSubForloop(op)) {
      CHECK(is_separated_);
      sub_forloop_counter_++;

      auto* min_n = op->extent.As<ir::Min>();
      CHECK(min_n);

      if (is_left_branch_) {  // the first
        node->extent = min_n->a;
      } else {  // the second
        node->extent = min_n->b;
      }
    } else {
      Visit(&node->body);
    }
  }

  //! Tell whether we meet the target subforloop, the forloop having the same iterator and min, extent in the body of
  //! the forloop(to separate) should be the target sub-forloop.
  bool MatchSubForloop(const ir::For* forloop) {
    return forloop->loop_var == sub_forloop_->loop_var && forloop->min == sub_forloop_->min &&
           forloop->extent == sub_forloop_->extent;
  }

 private:
  bool is_separated_{false};
  int sub_forloop_counter_;
  Expr* forloop_;
  Expr separator_;
  // Tell whether the forloop is located at the root's left branch(min, separator_).
  bool is_left_branch_{false};
  ir::For* sub_forloop_;
};

/*
 * Separate a forloop, if successfully found one and seperate it, just return.
 * NOTE The input expressions can only deal with PolyFor, not For nodes.
 */
struct ForAutoSeparateMutator : ir::IRMutator<Expr*> {
  Expr* operator()(Expr* expr) {
    ir::IRMutator<>::Visit(expr, expr);
    return separated_forloop;
  }

 private:
  //! Fill it if a forloop is separated.
  Expr* separated_forloop{};
  void Visit(Expr* expr) {
    // The root_ might be replaced only if root_ == the forloop_to_separate.
    ir::IRMutator<>::Visit(expr, expr);
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    forloop_stack.push_back(expr);

    do {  // We use a do-while here to break in any position and continue the post-procession after the while block.
      auto* min_n = op->extent.As<ir::Min>();
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
        auto* forloop = _forloop->As<ir::For>();
        bool contains =
            common::MathContainsSymbol(left, forloop->loop_var) || common::MathContainsSymbol(right, forloop->loop_var);
        if (contains) separate_levels.push_back(level);
        level++;
      }
      //! ignore the complex cases.
      if (separate_levels.size() > 1) break;
      CHECK_EQ(separate_levels.size(), 1UL);

      Expr* forloop_to_separate_expr = forloop_stack[separate_levels.front()];
      auto forloop_to_separate       = forloop_to_separate_expr->As<ir::For>();

      // check the min not include the current iterator, or it is illegal.
      Expr solve_res;
      bool is_positive;
      // solve left <= right
      std::tie(solve_res, is_positive) = common::Solve(right, left, forloop_to_separate->loop_var);
      VLOG(4) << "solve_res: " << solve_res;
      VLOG(4) << "is_positive: " << is_positive;

      // make a round if is a float
      if (solve_res.type().is_float()) {
        float v   = solve_res.as_float();
        int32_t x = is_positive ? std::floor(v) : std::ceil(v);
        solve_res = Expr(x);
      }

      // separate to two forloops with domain:
      // 1. (0, solve_res) with min.lhs
      // 2. (solve_res, extent) with min.rhs

      ForSeparater for_separater(forloop_to_separate_expr, solve_res, node);
      for_separater(forloop_to_separate_expr);

      separated_forloop = forloop_to_separate_expr;
      return;
      // Visit(forloop_to_separate_expr);
      // iterator >= solve_res
    } while (false);

    Visit(&node->body);
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

Expr* PolyForAutoSeparateHelper(Expr* expr) {
  ForAutoSeparateMutator mutator;
  return mutator(expr);
}

struct ForAutoSeparateMutatorMain : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { Visit(expr); }

 private:
  void Visit(const ir::Block* op, Expr* expr) {
    auto* node = expr->As<ir::Block>();
    for (auto& expr : node->stmts) {
      auto* res = PolyForAutoSeparateHelper(&expr);
      if (res) {
        Visit(res);
      }
    }
  }

  void Visit(const ir::For* op, Expr* expr) {
    auto* res = PolyForAutoSeparateHelper(expr);
    if (res) Visit(res);
  }

  void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }
};

Expr PlusOneWithMinMax(Expr expr) {
  auto* min_n = expr.As<ir::Min>();
  auto* max_n = expr.As<ir::Max>();

  if (min_n) {
    min_n->a = min_n->a + 1;
    min_n->b = min_n->b + 1;
    Simplify(&min_n->a);
    Simplify(&min_n->b);
    return expr;
  } else if (max_n) {
    max_n->a = max_n->a + 1;
    max_n->b = max_n->b + 1;
    Simplify(&max_n->a);
    Simplify(&max_n->b);
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

    Expr lhs = lt_n ? lt_n->a : le_n->a;
    Expr rhs = lt_n ? lt_n->b : PlusOneWithMinMax(le_n->b);
    if (common::IsPureMath(rhs)) Simplify(&rhs);

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

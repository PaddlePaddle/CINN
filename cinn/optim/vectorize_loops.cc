#include "cinn/optim/vectorize_loops.h"
#include <algorithm>
#include "cinn/ir/ir_printer.h"
#include "cinn/utils/functional.h"

namespace cinn {
namespace optim {
using namespace ir;

//! Substitutes a vector for a scalar var in a Stmt.
struct VectorSubs : public IRMutator<Expr*> {
  //! The name of the variable to be vectorized.
  std::string var;

  //! The expression to replace with, usually a ramp.
  Expr replacement;

  const Target& target;

  //! A suffix to attach to widened variables.
  std::string widen_suffix;

  VectorSubs(const std::string& v, Expr r, const Target& t) : var(v), replacement(r), target(t) {
    widen_suffix = ".x" + std::to_string(replacement.type().lanes());
  }

  //! Widen an expression to the given number of lanes.
  Expr Widen(Expr e, int lanes) {
    if (e.type().lanes() == lanes) {
      return e;
    } else if (e.type().lanes() == 1) {
      return Broadcast::Make(e, lanes);
    } else {
      LOG(FATAL) << "Mismatched vector lanes";
    }
    return Expr();
  }
  void Visit(Expr* expr) { IRMutator<Expr*>::Visit(expr, expr); }

  void Visit(const Cast* op, Expr* expr) override {
    auto* node = expr->As<Cast>();
    Visit(&node->v);

    Type t = op->type().with_lanes(node->v.type().lanes());
    node->set_type(t);
  }

  void Visit(const _Var_* op, Expr* expr) override {
    auto* node               = expr->As<_Var_>();
    std::string widened_name = op->name + widen_suffix;
    if (op->name == var) {
      *expr = replacement;
    }
  }

  void Visit(const Add* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const Sub* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const Mul* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const Div* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const Mod* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const Min* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const Max* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const EQ* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const NE* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const LT* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const LE* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const GT* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const GE* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const And* op, Expr* expr) override { MutateBinaryOperator(op, expr); }
  void Visit(const Or* op, Expr* expr) override { MutateBinaryOperator(op, expr); }

  void Visit(const Select* op, Expr* expr) override {
    auto* node = expr->As<Select>();
    Visit(&node->condition);
    Visit(&node->true_value);
    Visit(&node->false_value);

    int lanes =
        utils::Max(node->condition.type().lanes(), node->true_value.type().lanes(), node->false_value.type().lanes());
    node->true_value  = Widen(node->true_value, lanes);
    node->false_value = Widen(node->false_value, lanes);
  }

  void Visit(const Load* op, Expr* expr) override {
    auto* node = expr->As<Load>();
    // We ignore the predicate here.
    Visit(&node->index);

    int width = node->index.type().lanes();
    node->set_type(node->type().with_lanes(width));
  }

  void Visit(const Call* op, Expr* expr) override { LOG(ERROR) << "Ignore widen Call node"; }

  void Visit(const Let* op, Expr* expr) override {
    auto* node = expr->As<Let>();
    Visit(&node->value);
  }

  void Visit(const Store* op, Expr* expr) override {
    auto* node = expr->As<Store>();
    Visit(&node->value);
    Visit(&node->index);
    int lanes = std::max(node->value.type().lanes(), node->index.type().lanes());

    node->value = Widen(node->value, lanes);
    node->index = Widen(node->index, lanes);
  }

  void Visit(const IfThenElse* op, Expr* expr) override {
    auto* node = expr->As<IfThenElse>();
    Visit(&node->condition);
    int lanes = node->condition.type().lanes();
    Visit(&node->true_case);
    Visit(&node->false_case);
    LOG(ERROR) << "Ignore Width IfThenElse";
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node       = expr->As<PolyFor>();
    ForType for_type = op->for_type;
  }

  template <typename T>
  void MutateBinaryOperator(const T* op, Expr* expr) {
    auto* node = expr->As<T>();
    Visit(&node->a);
    Visit(&node->b);

    int width = std::max(node->a.type().lanes(), node->b.type().lanes());
    *expr     = T::Make(Widen(node->a, width), Widen(node->b, width));
  }
};

struct VectorizeLoops_ : public IRMutator<Expr*> {
  const Target& target;

  VectorizeLoops_(const Target& t) : target(t) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  void Visit(const PolyFor* forloop, Expr* expr) {
    auto* node = expr->As<PolyFor>();

    if (forloop->for_type == ForType::Vectorized) {
      // The forloop generated from polyhedral analysis might have a complex condition that is not something like "i<20"
      // or "i<=20", those cases is not possible to extract the extent.
      auto* extent_int = forloop->extent().As<IntImm>();
      if (!extent_int) {
        VLOG(2) << "Ignore the forloop because the condition is not based on a int extent";
        return;
      }

      int extent = extent_int->value;
      CHECK_GT(extent, 0) << "Loop over " << Expr(forloop->iterator) << " has extent " << forloop->extent()
                          << ". Can only vectorize loops over a constant extent > 1";

      Expr for_var     = _Var_::Make(forloop->iterator->name, Int(32));
      Expr replacement = Ramp::Make(forloop->init, Expr(1), extent);
      VectorSubs(forloop->iterator->name, replacement, target).Visit(&node->body);
    } else {
      IRMutator::Visit(forloop, expr);
    }
  }
};

void VectorizeLoops(Expr* expr, const Target& target) { return VectorizeLoops_(target)(expr); }

}  // namespace optim
}  // namespace cinn

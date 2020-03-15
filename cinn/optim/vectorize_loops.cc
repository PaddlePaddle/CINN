#include "cinn/optim/vectorize_loops.h"

#include <algorithm>
#include <string>

#include "cinn/common/ir.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/utils/functional.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using common::make_const;
using common::make_one;
using common::make_zero;

//! Widen an expression to the given number of lanes.
Expr Widen(Expr e, int lanes) {
  if (e.type().lanes() == lanes) return e;
  if (const ir::Broadcast* op = e.As<ir::Broadcast>()) {
    if (lanes % op->lanes == 0) {
      return ir::Broadcast::Make(op->value, lanes);
    }
  }

  CHECK_EQ(e.type().lanes(), 1) << "Cannot broadcast lanes from " << e.type().lanes() << " to " << lanes;
  return ir::Broadcast::Make(e, lanes);
}

//! Substitutes a vector for a scalar var in a Stmt.
class Vectorizer : public IRMutator<Expr*> {
  //! The name of the variable to be vectorized.
  Var var;

  int lanes_{-1};

  bool need_scalarize_{false};

  Expr ramp_;

  //! A suffix to attach to widened variables.
  std::string widen_suffix;

 public:
  Vectorizer(const Var& var, int lanes) : var(var), lanes_(lanes) {
    // the identity ramp.
    ramp_ = Ramp::Make(make_zero(), make_one(), lanes_);
  }

  void Visit(Expr* expr) {
    CHECK(!need_scalarize_);
    IRMutator<Expr*>::Visit(expr, expr);

    if (need_scalarize_) {
      need_scalarize_ = false;
      Scalarize(expr);
    }
  }

  void Visit(const Cast* op, Expr* expr) override {
    auto* node = expr->As<Cast>();
    auto v0    = node->v;
    Visit(&node->v);
    if (v0.same_as(node->v)) return;

    Type t = op->type().with_lanes(node->v.type().lanes());
    node->set_type(t);
  }

  void Visit(const _Var_* op, Expr* expr) override {
    if (op->name == var->name) {
      *expr = Expr(ramp_);
      return;
    }
  }

  void Visit(const Add* op, Expr* expr) override { MutateAddSubOperator(op, expr); }
  void Visit(const Sub* op, Expr* expr) override { MutateAddSubOperator(op, expr); }
  void Visit(const Mul* op, Expr* expr) override { MutateMulDivOperator(op, expr); }
  void Visit(const Div* op, Expr* expr) override { MutateMulDivOperator(op, expr); }
  void Visit(const Mod* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const Min* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const Max* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const EQ* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const NE* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const LT* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const LE* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const GT* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const GE* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const And* op, Expr* expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const Or* op, Expr* expr) override { BinaryOperatorVec(op, expr); }

  void Visit(const Ramp* op, Expr* expr) override {}

  void Visit(const Select* op, Expr* expr) override {
    auto* node        = expr->As<Select>();
    auto condition0   = node->condition;
    auto true_value0  = node->true_value;
    auto false_value0 = node->false_value;

    Visit(&node->condition);
    Visit(&node->true_value);
    Visit(&node->false_value);

    if (condition0.same_as(node->condition) && true_value0.same_as(node->true_value) &&
        false_value0.same_as(node->false_value))
      return;

    int lanes =
        utils::Max(node->condition.type().lanes(), node->true_value.type().lanes(), node->false_value.type().lanes());
    node->true_value  = Widen(node->true_value, lanes);
    node->false_value = Widen(node->false_value, lanes);
  }

  void Visit(const Load* op, Expr* expr) override {
    auto* node  = expr->As<Load>();
    auto index0 = node->index;
    // We ignore the predicate here.
    Visit(&node->index);
    if (index0.same_as(node->index)) return;

    int width = node->index.type().lanes();

    *expr = Load::Make(node->tensor, node->index);
  }

  void Visit(const Call* op, Expr* expr) override { LOG(ERROR) << "Ignore widen Call node"; }

  void Visit(const Let* op, Expr* expr) override {
    auto* node = expr->As<Let>();
    Visit(&node->value);
    LOG(ERROR) << "Let not supported";
  }

  void Visit(const Store* op, Expr* expr) override {
    auto* node  = expr->As<Store>();
    auto value0 = node->value;
    auto index0 = node->index;
    Visit(&node->value);
    Visit(&node->index);
    if (value0.same_as(node->value) && index0.same_as(node->index)) return;

    int lanes   = std::max(node->value.type().lanes(), node->index.type().lanes());
    node->value = Widen(node->value, lanes);
    node->index = Widen(node->index, lanes);

    *expr = Store::Make(node->tensor, node->value, node->index);
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
    NOT_IMPLEMENTED
  }

  void Scalarize(Expr* expr) {
    Var idx(var->name + "_s", Int(32));
    std::map<const ir::_Var_*, Expr> var_map;
    var_map[var.As<ir::_Var_>()] = idx;

    common::Substitute(expr, var_map);
    *expr =
        ir::For::Make(idx, common::make_const(0), common::make_const(lanes_), ForType::Serial, DeviceAPI::Host, *expr);
  }

  template <typename T>
  void MutateAddSubOperator(const T* op, Expr* expr) {
    auto* node = expr->As<T>();
    Expr a0    = node->a;
    Expr b0    = node->b;
    Visit(&node->a);
    Visit(&node->b);

    if (a0.same_as(node->a) && b0.same_as(node->b)) return;

    int lanes = std::max(node->a.type().lanes(), node->b.type().lanes());
    if (lanes != 1) {
      const Ramp* a_ramp_n = node->a.template As<Ramp>();
      const Ramp* b_ramp_n = node->b.template As<Ramp>();
      if (node->a.type().lanes() == 1 && b_ramp_n) {
        // a + Ramp(base,stride,lanes) = Ramp(base+a, stride,lanes)
        *expr = Ramp::Make(T::Make(node->a, b_ramp_n->base),  // base
                           b_ramp_n->stride,                  // stride
                           b_ramp_n->lanes);
        return;
      }
      if (node->b.type().lanes() == 1 && a_ramp_n) {
        *expr = Ramp::Make(T::Make(node->b, a_ramp_n->base),  // base
                           a_ramp_n->stride,                  // stride
                           a_ramp_n->lanes);
        return;
      }
    }

    *expr = T::Make(Widen(node->a, lanes), Widen(node->b, lanes));
  }

  template <typename T>
  void MutateMulDivOperator(const T* op, Expr* expr) {
    Expr a0    = op->a;
    Expr b0    = op->b;
    auto* node = expr->As<T>();
    Visit(&node->a);
    Visit(&node->b);

    if (a0.same_as(node->a) && b0.same_as(node->b)) return;

    int lanes = std::max(node->a.type().lanes(), node->b.type().lanes());
    if (lanes != 1) {
      const Ramp* a_ramp_n = node->a.template As<Ramp>();
      const Ramp* b_ramp_n = node->b.template As<Ramp>();
      if (node->a.type().lanes() == 1 && b_ramp_n) {
        // a * Ramp(base,stride,lanes) = Ramp(base*a, stride*a,lanes)
        *expr = Ramp::Make(T::Make(node->a, b_ramp_n->base),    // base
                           T::Make(node->a, b_ramp_n->stride),  // stride
                           b_ramp_n->lanes);
        return;
      }
      // Ramp(base,stride,lanes) * b  = Ramp(base*b, stride*b,lanes)
      if (node->b.type().lanes() == 1 && a_ramp_n) {
        *expr = Ramp::Make(T::Make(a_ramp_n->base, node->b),    // base
                           T::Make(a_ramp_n->stride, node->b),  // stride
                           a_ramp_n->lanes);
        return;
      }
    }

    *expr = T::Make(Widen(node->a, lanes), Widen(node->b, lanes));
  }

  template <typename T>
  Expr BinaryOperatorVec(const T* op, Expr* expr) {
    auto* node = expr->As<T>();
    Expr a0    = node->a;
    Expr b0    = node->b;
    Visit(&node->a);
    Visit(&node->b);
    if (a0.same_as(node->a) && b0.same_as(node->b)) return *expr;

    int lanes = std::max(node->a.type().lanes(), node->b.type().lanes());
    return T::Make(Widen(node->a, lanes), Widen(node->b, lanes));
  }
};

struct VectorizeLoops_ : public IRMutator<Expr*> {
  const Target& target;

  explicit VectorizeLoops_(const Target& t) : target(t) {}

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

      Vectorizer(forloop->iterator, extent).Visit(&node->body);
    } else {
      IRMutator::Visit(forloop, expr);
    }
  }
};

void VectorizeLoops(Expr* expr, const Target& target) { return VectorizeLoops_(target)(expr); }

namespace detail {

void Vectorize(Var var, int lanes, Expr* expr) {
  Vectorizer vectorizer(var, lanes);
  vectorizer.Visit(expr);
}

}  // namespace detail

}  // namespace optim
}  // namespace cinn

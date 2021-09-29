#include "cinn/optim/vectorize_loops.h"

#include <algorithm>
#include <map>
#include <string>
#include <absl/container/flat_hash_map.h>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_replace.h"
#include "cinn/optim/ir_simplify.h"
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
  if (const ir::Broadcast *op = e.As<ir::Broadcast>()) {
    if (lanes % op->lanes == 0) {
      return ir::Broadcast::Make(op->value, lanes);
    }
  }

  CHECK_EQ(e.type().lanes(), 1) << "Cannot broadcast lanes from " << e.type().lanes() << " to " << lanes;
  return ir::Broadcast::Make(e, lanes);
}

//! Substitutes a vector for a scalar var in a Stmt.
class Vectorizer : public IRMutator<Expr *> {
  //! The name of the variable to be vectorized.
  Var var;

  int lanes_{-1};

  bool need_scalarize_{false};

  bool to_vectorize_{false};

  Expr ramp_;

  absl::flat_hash_map<std::string, common::CasInterval> var_intervals_;

  //! A suffix to attach to widened variables.
  std::string widen_suffix;

 public:
  Vectorizer(const Var &var, int lanes, const absl::flat_hash_map<std::string, common::CasInterval> &var_intervals = {})
      : var(var), lanes_(lanes), var_intervals_(var_intervals) {
    // the identity ramp.
    ramp_ = Ramp::Make(make_zero(), make_one(), lanes_);
  }

  void Visit(Expr *expr) {
    CHECK(!need_scalarize_);
    IRMutator<Expr *>::Visit(expr, expr);

    if (need_scalarize_) {
      need_scalarize_ = false;
      Scalarize(expr);
    }
  }

  void Visit(const Cast *op, Expr *expr) override {
    auto *node = expr->As<Cast>();
    auto v0    = node->v();
    Visit(&node->v());
    if (v0.same_as(node->v())) return;

    Type t = op->type().with_lanes(node->v().type().lanes());
    node->set_type(t);
  }

  void Visit(const _Var_ *op, Expr *expr) override {
    if (op->name == var->name) {
      *expr = Expr(ramp_);
    }
  }

  void Visit(const Add *op, Expr *expr) override { MutateAddSubOperator(op, expr); }
  void Visit(const Sub *op, Expr *expr) override { MutateAddSubOperator(op, expr); }
  void Visit(const Mul *op, Expr *expr) override { MutateMulDivOperator(op, expr); }
  void Visit(const Div *op, Expr *expr) override { MutateMulDivOperator(op, expr); }
  void Visit(const Mod *op, Expr *expr) override { MutateMulDivOperator(op, expr); }
  void Visit(const Min *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const Max *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const EQ *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const NE *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const LT *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const LE *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const GT *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const GE *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const And *op, Expr *expr) override { BinaryOperatorVec(op, expr); }
  void Visit(const Or *op, Expr *expr) override { BinaryOperatorVec(op, expr); }

  void Visit(const Ramp *op, Expr *expr) override {}

  void Visit(const Select *op, Expr *expr) override {
    auto *node        = expr->As<Select>();
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

  void Visit(const Load *op, Expr *expr) override {
    auto *node                = expr->As<Load>();
    std::vector<Expr> indices = node->indices;
    // We ignore the predicate here.
    bool need_visit = false;
    for (int i = 0; i < indices.size(); i++) {
      Visit(&node->indices[i]);
      if (!node->indices[i].same_as(indices[i])) {
        need_visit = true;
      }
    }
    if (!need_visit) return;
    int lanes = 0;
    for (auto &idx : node->indices) {
      lanes = std::max(idx.type().lanes(), lanes);
    }
    std::vector<Expr> new_indices;
    for (auto &idx : node->indices) {
      new_indices.push_back(Widen(idx, lanes));
    }
    *expr = Load::Make(node->tensor, new_indices);
  }

  void Visit(const Store *op, Expr *expr) override {
    auto *node  = expr->As<Store>();
    auto value0 = node->value;
    Visit(&node->value);

    std::vector<Expr> indices = node->indices;
    // We ignore the predicate here.
    for (auto &idx : node->indices) {
      Visit(&idx);
    }

    bool need_visit = false;
    for (int i = 0; i < indices.size(); i++) {
      if (!node->indices[i].same_as(indices[i])) {
        need_visit = true;
      }
    }
    if (!need_visit) return;

    int lanes = 0;
    for (auto &idx : node->indices) lanes = std::max(idx.type().lanes(), lanes);
    lanes = std::max(lanes, node->value.type().lanes());

    node->value = Widen(node->value, lanes);

    std::vector<Expr> new_indices;
    for (auto &idx : node->indices) {
      new_indices.push_back(Widen(idx, lanes));
    }
    *expr = Store::Make(node->tensor, node->value, new_indices);
  }

  void Visit(const Call *op, Expr *expr) override {
    std::vector<Expr> read_args  = op->read_args;
    std::vector<Expr> write_args = op->write_args;
    auto *node                   = expr->As<Call>();
    ir::IRMutator<>::Visit(op, expr);
    bool is_changed = false;
    int lanes       = 0;
    for (int i = 0; i < node->read_args.size(); i++) {
      lanes = std::max(node->read_args[i].type().lanes(), lanes);
      if (!node->read_args[i].same_as(read_args[i])) {
        is_changed = true;
      }
    }
    for (int i = 0; i < node->write_args.size(); i++) {
      lanes = std::max(node->write_args[i].type().lanes(), lanes);
      if (!node->write_args[i].same_as(write_args[i])) {
        is_changed = true;
      }
    }
    if (!is_changed) return;

    for (int i = 0; i < read_args.size(); i++) {
      node->read_args[i] = Widen(node->read_args[i], lanes);
    }
    for (int i = 0; i < write_args.size(); i++) {
      node->write_args[i] = Widen(node->write_args[i], lanes);
    }

    CHECK(!read_args.empty());
    Type type = op->type().with_lanes(lanes);
    *expr     = Call::Make(type,
                       node->name,
                       node->read_args,
                       node->write_args,
                       node->call_type,
                       node->func,
                       node->value_index,
                       node->attrs);
  }

  void Visit(const Let *op, Expr *expr) override {
    auto *node = expr->As<Let>();
    Visit(&node->symbol);
    LOG(ERROR) << "Let not supported";
  }

  void Visit(const IfThenElse *op, Expr *expr) override {
    auto *node = expr->As<IfThenElse>();
    Visit(&node->condition);
    int lanes = node->condition.type().lanes();
    Visit(&node->true_case);
    Visit(&node->false_case);
    LOG(ERROR) << "Ignore Width IfThenElse";
  }

  void Visit(const For *op, Expr *expr) override { ir::IRMutator<>::Visit(op, expr); }

  void Scalarize(Expr *expr) {
    Var idx(var->name + "_s", Int(32));
    std::map<const ir::_Var_ *, Expr> var_map;
    var_map[var.As<ir::_Var_>()] = idx;

    common::Substitute(expr, var_map);
    *expr =
        ir::For::Make(idx, common::make_const(0), common::make_const(lanes_), ForType::Serial, DeviceAPI::Host, *expr);
  }

  template <typename T>
  void MutateAddSubOperator(const T *op, Expr *expr) {
    auto *node = expr->As<T>();
    Expr a0    = node->a();
    Expr b0    = node->b();

    Visit(&node->a());
    Visit(&node->b());

    // if (a0.same_as(node->a()) && b0.same_as(node->b())) return;

    int lanes = std::max(node->a().type().lanes(), node->b().type().lanes());
    if (lanes != 1) {
      const Ramp *a_ramp_n = node->a().template As<Ramp>();
      const Ramp *b_ramp_n = node->b().template As<Ramp>();
      if (node->a().type().lanes() == 1 && b_ramp_n) {
        // a + Ramp(base,stride,lanes) = Ramp(base+a, stride,lanes)
        *expr = Ramp::Make(T::Make(node->a(), b_ramp_n->base),  // base
                           b_ramp_n->stride,                    // stride
                           b_ramp_n->lanes);
        return;
      }
      if (node->b().type().lanes() == 1 && a_ramp_n) {
        *expr = Ramp::Make(T::Make(node->b(), a_ramp_n->base),  // base
                           a_ramp_n->stride,                    // stride
                           a_ramp_n->lanes);
        return;
      }
    }

    *expr = T::Make(Widen(node->a(), lanes), Widen(node->b(), lanes));
  }

  template <typename T>
  void MutateMulDivOperator(const T *op, Expr *expr) {
    Expr a0    = op->a();
    Expr b0    = op->b();
    auto *node = expr->As<T>();
    Visit(&node->a());
    Visit(&node->b());

    // if (a0.same_as(node->a()) && b0.same_as(node->b())) return;
    int lanes = std::max(node->a().type().lanes(), node->b().type().lanes());
    if (lanes != 1) {
      const Ramp *a_ramp_n = node->a().template As<Ramp>();
      const Ramp *b_ramp_n = node->b().template As<Ramp>();
      if (node->a().type().lanes() == 1 && b_ramp_n) {
        // a * Ramp(base,stride,lanes) = Ramp(base*a, stride*a,lanes)
        *expr = Ramp::Make(T::Make(node->a(), b_ramp_n->base),    // base
                           T::Make(node->a(), b_ramp_n->stride),  // stride
                           b_ramp_n->lanes);

        return;
      }
      // Ramp(base,stride,lanes) * b  = Ramp(base*b, stride*b,lanes)
      if (node->b().type().lanes() == 1 && a_ramp_n) {
        *expr = Ramp::Make(T::Make(a_ramp_n->base, node->b()),    // base
                           T::Make(a_ramp_n->stride, node->b()),  // stride
                           a_ramp_n->lanes);
        return;
      }
    }

    *expr = T::Make(Widen(node->a(), lanes), Widen(node->b(), lanes));
  }

  template <typename T>
  void BinaryOperatorVec(const T *op, Expr *expr) {
    auto *node = expr->As<T>();
    Expr a0    = node->a();
    Expr b0    = node->b();
    Visit(&node->a());
    Visit(&node->b());
    // if (a0.same_as(node->a()) && b0.same_as(node->b())) return *expr;

    int lanes = std::max(node->a().type().lanes(), node->b().type().lanes());
    *expr     = T::Make(Widen(node->a(), lanes), Widen(node->b(), lanes));
  }
};

struct VectorizeLoops_ : public IRMutator<Expr *> {
  const Target &target;
  absl::flat_hash_map<std::string, common::CasInterval> var_intervals;
  bool vectorizable_ = true;

  explicit VectorizeLoops_(const Target &t) : target(t) {}

  void operator()(Expr *expr) { IRMutator::Visit(expr, expr); }

  void Visit(const Load *op, Expr *expr) override {
    auto *node                = expr->As<Load>();
    std::vector<Expr> indices = node->indices;

    bool is_changed = false;
    // simplify the complicated index from poly in the format of div/mod
    for (int i = 0; i < indices.size(); i++) {
      node->indices[i] = common::AutoSimplify(node->indices[i], var_intervals);
      Simplify(&node->indices[i]);
      if (!node->indices[i].same_as(indices[i])) {
        is_changed = true;
      }
    }
    if (!is_changed) return;

    *expr = Load::Make(node->tensor, node->indices);
  }

  void Visit(const Store *op, Expr *expr) override {
    auto *node = expr->As<Store>();
    auto value = node->value;
    IRMutator::Visit(&node->value, &node->value);

    std::vector<Expr> indices = node->indices;
    bool is_changed           = false;
    // simplify the complicated index from poly in the format of div/mod
    for (int i = 0; i < indices.size(); i++) {
      node->indices[i] = common::AutoSimplify(node->indices[i], var_intervals);
      Simplify(&node->indices[i]);
      if (!node->indices[i].same_as(indices[i])) {
        is_changed = true;
      }
    }
    if (!is_changed) return;

    *expr = Store::Make(node->tensor, node->value, node->indices);
  }

  void Visit(const Call *op, Expr *expr) override {
    auto it = op->attrs.find("vectorizable");
    if (it != op->attrs.end()) {
      vectorizable_ = absl::get<bool>(it->second);
    }
  }

  void Visit(const For *forloop, Expr *expr) {
    auto *node        = expr->As<For>();
    auto loopvar_name = forloop->loop_var->name;
    if (forloop->extent.As<IntImm>()) {
      var_intervals.emplace(loopvar_name, common::CasInterval{0, forloop->extent.as_int32() - 1});
    } else {
      var_intervals.emplace(loopvar_name, common::CasInterval{Expr(0), forloop->extent - 1});
    }
    // the extent the forloops marked as Vectorized should be int constant
    if (forloop->is_vectorized()) {
      Context::Global().info_rgt().Get<int>("vectorized_forloop_count")++;

      CHECK(forloop->vectorize_info().valid());

      CHECK(is_zero(forloop->min));
      Expr for_extent = common::AutoSimplify(forloop->extent);
      Simplify(&for_extent);
      node->extent     = for_extent;
      auto *extent_min = for_extent.As<Min>();
      auto *extent_max = for_extent.As<Max>();

      vectorizable_ = true;
      IRMutator<>::Visit(&node->body, &node->body);
      if (extent_min || extent_max || !vectorizable_) {
        // not vectorize if has tail blocks, for llvm to optimize
        node->reset_vectorize_info();
        var_intervals.erase(forloop->loop_var->name);
        return;
      }

      auto _new_forloop = SplitForLoop(node, forloop->vectorize_info().factor);
      if (!_new_forloop.defined()) {
        IRMutator<>::Visit(&node->body, &node->body);
        var_intervals.erase(forloop->loop_var->name);
        return;
      }

      node->reset_vectorize_info();

      auto *new_forloop = _new_forloop.As<ir::For>();

      // The forloop generated from polyhedral analysis might have a complex condition that is not something like
      // "i<20" or "i<=20", those cases is not possible to extract the extent.
      auto *extent_int = new_forloop->extent.As<IntImm>();

      if (!extent_int) {
        IRMutator<>::Visit(&node->body, &node->body);
        var_intervals.erase(forloop->loop_var->name);
        return;
      }

      int extent = extent_int->value;
      CHECK_GT(extent, 0) << "Loop over " << Expr(new_forloop->loop_var) << " has extent " << new_forloop->extent
                          << ". Can only vectorize loops over a constant extent > 1";

      VLOG(2) << "Vectorizing " << new_forloop->loop_var << " extent " << extent;
      VLOG(2) << "body:\n" << node->body;

      Vectorizer(new_forloop->loop_var, extent, var_intervals).Visit(&new_forloop->body);

      VLOG(2) << "after vectorize body:\n" << node->body;

      // Remove the forloop, the new_forloop's body is vectorized to Ramp, so no forloop is needed.
      if (is_zero(forloop->extent - 1)) {
        *expr = new_forloop->body;
      } else {
        node->body = new_forloop->body;
      }
    } else {
      IRMutator::Visit(forloop, expr);
    }
    var_intervals.erase(loopvar_name);
  }

  //! unroll the forloop if its' extent is min type by solving the condition extent
  //! @return The new forloop.
  bool UnrollCmpFor(For *outer_for, For *inner_for, Expr *expr) {
    CHECK(outer_for);
    CHECK(inner_for);
    Expr inner_for_extent = common::AutoSimplify(inner_for->extent);
    Simplify(&inner_for_extent);
    auto *extent_min = inner_for_extent.As<Min>();
    if (extent_min) {
      CHECK(is_zero(inner_for->min));
      // simplify the complicated indices of load/store from poly
      IRMutator::Visit(&inner_for->body, &inner_for->body);
      Expr a, b, condition;
      a          = extent_min->a();
      b          = extent_min->b();
      auto a_int = a.As<IntImm>();
      auto b_int = a.As<IntImm>();
      if (a_int || b_int) {
        condition = common::SolveInequality(LE::Make(a, b), outer_for->loop_var);
        Simplify(&condition);
      }
      if (condition.defined()) {
        auto le_n      = condition.As<ir::LE>();
        bool can_split = le_n && le_n->b().is_constant();
        if (le_n && le_n->b().is_constant()) {
          Expr inner_for_a  = Block::Make({For::Make(inner_for->loop_var,
                                                    inner_for->min,
                                                    a,
                                                    ForType::Vectorized,
                                                    DeviceAPI::UNK,
                                                    inner_for->body,
                                                    inner_for->vectorize_info())});
          Expr new_extent_a = common::AutoSimplify(le_n->b() + 1);
          Expr out_for_a    = For::Make(outer_for->loop_var,
                                     outer_for->min,
                                     new_extent_a,
                                     outer_for->for_type(),
                                     outer_for->device_api,
                                     inner_for_a,
                                     outer_for->vectorize_info());
          Var new_iterator_inner(common::UniqName(inner_for->loop_var->name + "_s"));
          Var new_iterator_outer(common::UniqName(outer_for->loop_var->name + "_s"));

          Expr inner_for_b = Block::Make({For::Make(
              new_iterator_inner, inner_for->min, b, ForType::Serial, DeviceAPI::UNK, IRCopy(inner_for->body))});
          optim::IrReplace(&inner_for_b, inner_for->loop_var, Expr(new_iterator_inner));

          Expr out_for_b = For::Make(new_iterator_outer,
                                     new_extent_a,
                                     outer_for->extent,
                                     outer_for->for_type(),
                                     outer_for->device_api,
                                     inner_for_b,
                                     outer_for->vectorize_info());
          optim::IrReplace(&out_for_b, outer_for->loop_var, Expr(new_iterator_outer));
          *expr = Block::Make({out_for_a, out_for_b});
          VLOG(2) << *expr;
          IRMutator::Visit(expr, expr);
          return true;
        }
      }
    }
    return false;
  }

  //! Split the forloop with size \p factor.
  //! @return The new forloop.
  Expr SplitForLoop(For *forloop, int factor) {
    CHECK_GT(factor, 1);
    auto *for_min_i = forloop->min.As<IntImm>();
    CHECK(forloop);
    if (!for_min_i) return Expr();
    if (for_min_i->value != 0) return Expr();

    auto *extent_ptr = forloop->extent.As<IntImm>();
    Expr times;
    if (extent_ptr) {
      int extent_int   = forloop->extent.as_int32();
      int extent_trunc = extent_int / factor;
      int extent_times = extent_int % factor == 0 ? extent_trunc : extent_trunc + 1;
      times            = common::make_const(forloop->extent->type(), extent_times);
    } else {
      times = common::AutoSimplify(Div::Make(forloop->extent, make_const(factor)));
      Simplify(&times);
    }

    // update the current forloop
    auto times_int = times.As<IntImm>();
    forloop->set_vectorized(false);

    forloop->extent = times;
    if (times_int && forloop->extent.as_int32() >= 1) {
      var_intervals.emplace(forloop->loop_var->name, common::CasInterval{0, forloop->extent.as_int32() - 1});
    } else {
      var_intervals.erase(forloop->loop_var->name);
      var_intervals.emplace(forloop->loop_var->name, common::CasInterval{Expr(0), forloop->extent - 1});
    }

    // create the new forloop
    {
      Var new_iterator(Context::Global().NewName("vi"));
      var_intervals.emplace(new_iterator->name, common::CasInterval{0, factor - 1});
      // eliminate for 1
      Expr new_index;
      if (common::is_zero(times - 1)) {
        new_index = Expr(new_iterator);
      } else {
        new_index = Expr(forloop->loop_var) * factor + Expr(new_iterator);
      }
      optim::IrReplace(&forloop->body, forloop->loop_var, new_index);
      auto new_forloop = For::Make(new_iterator,
                                   forloop->min,
                                   make_const(factor),
                                   ForType::Vectorized,
                                   DeviceAPI::UNK,
                                   forloop->body,
                                   forloop->vectorize_info());
      forloop->body    = Block::Make({new_forloop});
      return new_forloop;
    }
  }
};

void VectorizeLoops(Expr *expr, const Target &target) { return VectorizeLoops_(target)(expr); }

namespace detail {

void Vectorize(Var var, int lanes, Expr *expr) {
  Vectorizer vectorizer(var, lanes);
  vectorizer.Visit(expr);
}

}  // namespace detail

}  // namespace optim
}  // namespace cinn

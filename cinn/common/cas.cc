#include "cinn/common/cas.h"

#include <algorithm>
#include <cmath>

#include "cinn/common/arithmatic.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace common {
using namespace ir;  // NOLINT

Expr AutoSimplify(Expr u, const std::unordered_map<std::string, CasInterval>& var_intervals) {
  u = detail::ConvertCinnToCAS(u);
  std::unordered_map<std::string, CasInterval> s_var_intervals;
  for (auto& item : var_intervals) {
    if (item.second.e_l.defined() && item.second.e_r.defined()) {
      Expr e_l = detail::ConvertCinnToCAS(item.second.e_l);
      Expr e_r = detail::ConvertCinnToCAS(item.second.e_r);
      s_var_intervals.emplace(item.first, CasInterval(e_l, e_r));
    } else {
      s_var_intervals.emplace(item.first, CasInterval(item.second.l, item.second.r));
    }
  }
  u = CasSimplify(u, s_var_intervals);
  u = detail::ConvertCasToCinn(u);
  return u;
}

int gcd(int a, int b) {
  // Everything divides 0
  if (a == 0) return b;
  if (b == 0) return a;
  if (a == 1 || b == 1) return 1;
  if (a < 0 || b < 0) {
    return gcd(std::abs(a), std::abs(b));
  }

  // base case
  if (a == b) return a;

  // a is greater
  if (a > b) return gcd(a - b, b);
  return gcd(a, b - a);
}

//////// All the following symbolic computation methods are implemented referencing to the book <Computer Algegra and
/// Symbolic Computation - Joel S. Cohen>

template <typename T>
std::vector<T> Rest(const std::vector<T>& vs) {
  return std::vector<T>(vs.begin() + 1, vs.end());
}

template <typename T>
std::vector<T> Concat(const std::vector<T>& as, const std::vector<T>& bs) {
  auto res = as;
  res.insert(std::end(res), bs.begin(), bs.end());
  return res;
}

// 3*x => 3
// x => 1
Expr ProductGetConstantPart(Expr u) {
  auto* product = u.As<Product>();
  if (product) {
    if (product->operand(0).is_constant()) {
      return product->operand(0);
    }
  }
  return make_const(u->type(), 1);
}

// 3*x => x
// x => x
// x^-1 => x^-1
Expr ProductGetNonConstantPart(Expr u) {
  auto* product = u.As<Product>();
  if (product) {
    if (product->operands().size() == 1) {
      auto a = product->operands().front();
      if (a.is_constant()) return Expr();
      return a;
    }

    if (product->operand(0).is_constant()) {
      if (product->operands().size() == 2) return product->operands()[1];
      auto rest = Rest(product->operands());
      if (rest.size() > 1) return Product::Make(Rest(product->operands()));
      return rest.front();
    }
  }
  return u;
}

Expr Base(Expr v) {
  auto* power_n = v.As<Power>();
  if (power_n) {
    return power_n->a();
  }
  return v;
}

Expr Exponent(Expr v) {
  auto* power_n = v.As<Power>();
  if (power_n) {
    return power_n->b();
  }
  return Expr(1);
}

namespace detail {

// Is a Divisiable to b.
// @{
bool IsDivisible(int64_t a, int64_t b) {
  CHECK_NE(b, 0);
  return a % b == 0;
}
bool IsDivisible(const Sum* a, int b);
bool IsDivisible(const Product* a, int b) {
  for (auto& item : a->operands()) {
    if (item.As<IntImm>() && IsDivisible(item.As<IntImm>()->value, b)) return true;
    if (item.As<Sum>() && IsDivisible(item.As<Sum>(), b)) return true;
  }
  return false;
}
bool IsDivisible(const Sum* a, int b) {
  for (auto& item : a->operands()) {
    auto* vi = item.As<IntImm>();
    auto* vp = item.As<Product>();
    if (vi && IsDivisible(vi->value, b)) continue;
    if (vp && IsDivisible(vp, b)) continue;
    return false;
  }
  return true;
}
bool IsDivisible(Expr a, int b) {
  auto* ai = a.As<IntImm>();
  auto* as = a.As<Sum>();
  auto* ap = a.As<Product>();

  if (ai) return IsDivisible(ai->value, b);
  if (as) return IsDivisible(as, b);
  if (ap) return IsDivisible(ap, b);
  return false;
}
// @}

//! Divide a by b, NOTE that a should be divisible by b.
// @{
Expr Divide(const Product* a, int b);
Expr Divide(const Sum* a, int b) {
  std::vector<Expr> args;
  for (auto& item : a->operands()) {
    if (item.As<IntImm>())
      args.push_back(make_const(item.type(), item.As<IntImm>()->value / b));
    else if (item.As<Product>())
      args.push_back(Divide(item.As<Product>(), b));
    else
      CINN_NOT_IMPLEMENTED
  }
  return Sum::Make(args);
}
Expr Divide(const Product* a, int b) {
  auto* a_first_i = a->operand(0).As<IntImm>();
  CHECK(a_first_i);
  int times = a_first_i->value / b;
  if (times == 1) {
    return Product::Make(Rest(a->operands()));
  } else {
    auto args = Rest(a->operands());
    args.insert(std::begin(args), make_const(a->type(), times));
    return Product::Make(args);
  }
}
// @}

inline int Iquot(int n, int d) { return n / d; }

inline int Irem(int n, int d) {
  int k = Iquot(n, d);
  return n - d * k;
}

Expr CasSimplifyMutator::SimplifyRationalNumber(Expr u) {
  auto* frac_n = u.As<FracOp>();
  if (frac_n) {
    Expr n = frac_n->a();
    Expr d = frac_n->b();

    auto* ni = n.As<IntImm>();
    auto* di = d.As<IntImm>();

    CHECK(ni && di);
    int nv = ni->value;
    int dv = di->value;

    if (Irem(nv, dv) == 0) {
      return Expr(make_const(u.type(), Iquot(nv, dv)));
    } else {
      int g = gcd(nv, dv);
      if (dv > 0) {
        return FracOp::Make(make_const(Iquot(nv, g)), make_const(Iquot(dv, g)));
      } else {
        return FracOp::Make(make_const(Iquot(-nv, g)), make_const(Iquot(-dv, g)));
      }
    }
  }
  return u;
}

Expr CasSimplifyMutator::SimplifyIntegerPower(Expr u) {
  auto* node = u.As<Power>();
  CHECK(node);
  Expr v = node->a();
  Expr n = node->b();
  CHECK(n.type().is_int());

  auto* vi = v.As<IntImm>();
  auto* vf = v.As<FloatImm>();
  auto* ni = n.As<IntImm>();
  if (vi) {
    if (vi->value == 0) return make_const(0);
    if (vi->value == 1) return make_const(1);
  }
  if (vf) {
    if (vf->value == 0.f) return make_const(vf->type(), 0.f);
    if (vf->value == 1.f) return make_const(vf->type(), 1.f);
  }
  if (ni) {
    // x^0 = 1
    if (ni->value == 0) {
      return make_const(u.type(), 1);
    }
    if (ni->value == 1) {
      return v;
    }
  }

  // 3 ^ k, k > 0, evaluate it.
  if (v.is_constant() && n.is_constant() && n.get_constant() > 0) {
    auto* vi = v.As<IntImm>();
    auto* ni = n.As<IntImm>();
    CHECK(vi && ni);
    return make_const(vi->type().ElementOf(), std::pow(vi->value, ni->value));
  }

  // x^a^b) -> x^(ab)
  auto* vp = v.As<Power>();
  if (vp) {
    Expr r = vp->a();
    Expr s = vp->b();
    Expr p = SimplifyProduct(Product::Make({n, s}));
    if (p.As<IntImm>()) {
      return SimplifyIntegerPower(Power::Make(r, p));
    } else {
      return Power::Make(r, p);
    }
  }

  return u;
}

Expr EvaluateConstantPower(Expr u) {
  auto* op = u.As<Power>();
  CHECK(op->b().type().is_int());

  auto* ai = op->a().As<IntImm>();
  auto* af = op->a().As<FloatImm>();
  auto* bi = op->b().As<IntImm>();

  if (ai && bi && bi->value < 0) return Expr();

  if (ai && bi) {
    return make_const(ai->type(), std::pow(ai->value, bi->value));
  }
  if (af && bi) {
    return make_const(af->type(), std::pow(af->value, bi->value));
  }

  return Expr();
}

Expr CasSimplifyMutator::SimplifyPower(Expr u) {
  auto* node = u.As<Power>();
  CHECK(node);
  Expr a = CasSimplify(node->a(), var_intervals);
  Expr b = CasSimplify(node->b(), var_intervals);

  {  // Evaluate
    auto tmp = EvaluateConstantPower(u);
    if (tmp.defined()) return tmp;
  }

  auto* int_v   = a.As<IntImm>();
  auto* float_v = a.As<FloatImm>();

  // 0^x = 0
  if ((int_v && int_v->value == 0) || (float_v && float_v->value == 0)) {
    return make_const(node->a().type(), 0);
  }
  // 1^x = 1
  if ((int_v && int_v->value == 1) || (float_v && float_v->value == 1.f)) {
    return make_const(node->a().type(), 1);
  }

  if (b.As<IntImm>()) {
    return SimplifyIntegerPower(Power::Make(a, b));
  }

  return u;
}

Expr SumOrProductGetSingleElementsRec(Expr u) {
  auto* product = u.As<Product>();
  auto* sum     = u.As<Sum>();
  if (product && product->operands().size() == 1) {
    return SumOrProductGetSingleElementsRec(u->operands.front());
  }
  if (sum && sum->operands().size() == 1) {
    return SumOrProductGetSingleElementsRec(u->operands.front());
  }
  return u;
}

double EvaluatePower(Expr u) {
  auto* power = u.As<Power>();
  auto a      = power->a();
  auto b      = power->b();

  auto bi = b.As<IntImm>();
  CHECK(bi);

  return std::pow(power->a().get_constant(), bi->value);
}

// Order, reference to Page 85.
bool ExprPosCmp::operator()(const Expr& a, const Expr& b) {
  // O-1, 1 <| 2
  if (a.is_constant() && b.is_constant()) {
    return a.get_constant() < b.get_constant();
  }
  if (a.As<Power>() && a.As<Power>()->is_constant() && b.is_constant()) {
    return EvaluatePower(a) < b.get_constant();
  }
  if (a.As<Power>() && a.As<Power>()->is_constant() && b.As<Power>() && b.As<Power>()->is_constant()) {
    return EvaluatePower(a) < EvaluatePower(b);
  }
  if (b.As<Power>() && b.As<Power>()->is_constant() && a.is_constant()) {
    return a.get_constant() < EvaluatePower(b);
  }

  // O-2, both are symbols, compare by the lexicographical order.
  if (a.As<_Var_>() && b.As<_Var_>()) {
    return a.As<_Var_>()->name < b.As<_Var_>()->name;
  }

  // O-3, if a and b are either both products or both sums, compare by each element similar to lexicographical order.
  if ((a.As<Product>() && b.As<Product>()) || (a.As<Add>() && b.As<Add>())) {
    auto& aoprs = a->operands;
    auto& boprs = b->operands;
    int m       = std::min(aoprs.size(), boprs.size());

    for (int i = 0; i < m; i++) {
      // ugly compare representation in string.
      auto& aopr = aoprs[aoprs.size() - 1 - i];
      auto& bopr = boprs[boprs.size() - 1 - i];
      if (aopr != bopr) return operator()(aopr, bopr);
    }

    return aoprs.size() < boprs.size();
  }

  // O-4, if both are powers
  {
    auto* ap = a.As<Power>();
    auto* bp = b.As<Power>();
    if (ap && bp) {
      // compare base
      if (ap->a() != bp->a()) {
        return operator()(ap->a(), bp->a());
      }
      // compare exponent
      return operator()(ap->b(), bp->b());
    }
  }

  // customized case, if both are mod
  {
    auto* am = a.As<Mod>();
    auto* bm = b.As<Mod>();
    if (am && bm) {
      if (am->b() != bm->b()) {
        return operator()(am->b(), bm->b());
      }
      return operator()(am->a(), bm->a());
    }
  }

  // O-7, if a is an integer or fraction and v is any other type, 1 < x
  if (a.As<IntImm>() || a.As<FloatImm>() || a.As<FracOp>()) {
    if (!(b.As<IntImm>() || b.As<FloatImm>() || b.As<FracOp>())) return true;
  }
  if (b.As<IntImm>() || b.As<FloatImm>() || b.As<FracOp>()) {
    if (!(a.As<IntImm>() || a.As<FloatImm>() || a.As<FracOp>())) return false;
  }

  // O-8, if a is a product, v is a power, sum, fractional, or symbol
  {
    auto* ap = a.As<Product>();

    if (ap && (b.As<Power>() || b.As<Sum>() || b.As<Call>() || b.As<_Var_>() || b.As<Mod>())) {
      return operator()(a, Product::Make({b}));
    }
  }

  // O-9, if a is a power, b is a sum, function or symbol
  {
    if (a.As<Power>()) {
      if (b.As<Add>() || b.As<Call>() || b.As<_Var_>() || b.As<Mod>() || b.As<Call>()) {
        return operator()(a, Power::Make(b, make_const(1)));
      }
    }
  }

  {
    if (a.As<Mod>()) {
      if (!b.As<Mod>()) {
        // Todo: may be wrong especially for negative value
        return operator()(a, Mod::Make(b, Sum::Make({b, Expr(1)})));
      }
    }
  }

  // O-10, if a is a sum, b is a function, or symbol
  {
    if (a.As<Sum>()) {
      if (b.As<_Var_>()) {
        return operator()(a, Sum::Make({b}));
      }
    }
  }

  return false;
}

std::vector<Expr> CasSimplifyMutator::MergeProduct(const std::vector<Expr>& _p, const std::vector<Expr>& _q) {
  std::vector<Expr> p, q;
  for (auto& e : _p) p.push_back(CasSimplify(e, var_intervals));
  for (auto& e : _q) q.push_back(CasSimplify(e, var_intervals));

  // MPRD-1,2
  if (p.empty()) return q;
  if (q.empty()) return p;

  // MPRD-3
  auto& p1 = p[0];
  auto& q1 = q[0];
  auto h   = SimplifyProductRec({p1, q1});

  // case 1
  if (h.empty()) {
    return MergeProduct(Rest(p), Rest(q));
  }

  // case 2
  if (h.size() == 1) {
    auto rest = MergeProduct(Rest(p), Rest(q));
    rest.insert(std::begin(rest), h[0]);
    return rest;
  }

  // case 3
  if (h.size() == 2 && h[0] == p1 && h[1] == q1) {
    auto rest = MergeProduct(Rest(p), q);
    rest.insert(std::begin(rest), p1);
    return rest;
  }

  // case 4
  if (h.size() == 2 && h[0] == q1 && h[1] == p1) {
    auto rest = MergeProduct(p, Rest(q));
    rest.insert(std::begin(rest), q1);
    return rest;
  }

  // rest
  return Concat(p, q);
}

std::vector<Expr> CasSimplifyMutator::CasSimplifyMutator::SimplifyProductRec(const std::vector<Expr>& _operands) {
  CHECK_GE(_operands.size(), 2);
  std::vector<Expr> operands;
  for (auto& e : _operands) operands.push_back(CasSimplify(e, var_intervals));

  // SPRDREC-1
  if (operands.size() == 2 && !operands[0].As<Product>() && !operands[1].As<Product>()) {
    auto a = operands[0];
    auto b = operands[1];

    auto* ai = a.As<IntImm>();
    auto* af = a.As<FloatImm>();
    auto* bi = b.As<IntImm>();
    auto* bf = b.As<FloatImm>();

    // case 1, both are constants
    if (a.is_constant() && b.is_constant()) {
      if (ai) return {make_const(a.type(), ai->value * bi->value)};
      if (af) return {make_const(a.type(), af->value * bf->value)};
    }

    if (a.As<Max>() || a.As<Min>() || b.As<Max>() || b.As<Min>()) {
      // cinn_min/cinn_max(a, b) * 2 = cinn_min/cinn_max(2*a, 2*b)
      // 2 * cinn_min/cinn_max(a, b) = cinn_min/cinn_max(2*a, 2*b)
      // cinn_min/cinn_max(a, b) * -2 = cinn_max/cinn_min(-2*b, -2*a)
      // -2 * cinn_min/cinn_max(a, b) = cinn_max/cinn_min(-2*b, -2*a)
      Expr const_oper;
      Expr cmp_oper;
      int const_value;
      if (ai) {
        const_oper  = a;
        cmp_oper    = b;
        const_value = ai->value;
      }
      if (af) {
        const_oper  = a;
        cmp_oper    = b;
        const_value = af->value;
      }
      if (bi) {
        const_oper  = b;
        cmp_oper    = a;
        const_value = bi->value;
      }
      if (bf) {
        const_oper  = b;
        cmp_oper    = a;
        const_value = bf->value;
      }
      if (const_value == 0) {
        return {make_const(a->type(), 0)};
      }
      if (cmp_oper.defined() && const_oper.defined()) {
        auto cmp_min = cmp_oper.As<Min>();
        auto cmp_max = cmp_oper.As<Max>();
        if (const_value > 0) {
          if (cmp_min) {
            return {CasSimplify(Min::Make(CasSimplify(Product::Make({cmp_min->a(), const_oper}), var_intervals),
                                          CasSimplify(Product::Make({cmp_min->b(), const_oper}), var_intervals)),
                                var_intervals)};
          }
          if (cmp_max) {
            return {CasSimplify(Max::Make(CasSimplify(Product::Make({cmp_max->a(), const_oper}), var_intervals),
                                          CasSimplify(Product::Make({cmp_max->b(), const_oper}), var_intervals)),
                                var_intervals)};
          }
        } else {
          if (cmp_min) {
            return {CasSimplify(Max::Make(CasSimplify(Product::Make({cmp_min->b(), const_oper}), var_intervals),
                                          CasSimplify(Product::Make({cmp_min->a(), const_oper}), var_intervals)),
                                var_intervals)};
          }
          if (cmp_max) {
            return {CasSimplify(Min::Make(CasSimplify(Product::Make({cmp_max->b(), const_oper}), var_intervals),
                                          CasSimplify(Product::Make({cmp_max->a(), const_oper}), var_intervals)),
                                var_intervals)};
          }
        }
      }
    }

    {  // FracOp related constants.
      // NOTE the integer division is weried in C language, 1/2 = 0, that is huge different from a real CAS.
      auto* af = a.As<FracOp>();
      auto* bf = b.As<FracOp>();
      // 1/2 * 2/3
      if (af && bf && a->type().is_float()) {
        return {CasSimplify(FracOp::Make(Product::Make({af->a(), bf->a()}), Product::Make({af->b(), bf->b()})),
                            var_intervals)};
      }
      if (af && !bf && a->type().is_float()) {
        return {CasSimplify(FracOp::Make(Product::Make({af->a(), b}), af->b()), var_intervals)};
      }
      if (!af && bf && a->type().is_float()) {
        return {CasSimplify(FracOp::Make(Product::Make({bf->a(), a}), bf->b()), var_intervals)};
      }
    }

    // case 2
    // x*1 -> a
    if (ai && ai->value == 1) return {b};
    if (af && af->value == 1.f) return {b};
    // 1*x -> x
    if (bi && bi->value == 1) return {a};
    if (bf && bf->value == 1.f) return {a};

    // case 3
    if (Base(a) == Base(b)) {
      Expr s = SimplifySum(Sum::Make({Exponent(a), Exponent(b)}));
      Expr p = SimplifyPower(Power::Make(Base(a), s));
      return {p};
    }

    {  // power related constants
      auto* ap = a.As<Power>();
      auto* bp = b.As<Power>();

      auto one_is_power = [this](Expr _a, Expr _b) -> std::vector<Expr> {
        auto* ap = _a.As<Power>();
        auto* bp = _b.As<Power>();
        auto* bi = _b.As<IntImm>();

        CHECK(ap);
        CHECK(!bp);
        auto* ap_base_i = ap->a().As<IntImm>();
        CHECK(ap_base_i);  // if is float, it should be evaluated to a number.
        auto* ap_exponent_i = ap->b().As<IntImm>();
        CHECK(ap_exponent_i) << "exponent of a power should be an integer";
        CHECK_EQ(ap_exponent_i->value, -1);  // or it should be evaluated to a constant.
        if (bi) {
          int g       = gcd(ap_base_i->value, bi->value);
          int base    = ap_base_i->value / g;
          int b_value = bi->value / g;
          auto a_new  = Power::Make(make_const(ap->a().type(), base), make_const(-1));
          auto b_new  = make_const(_b.type(), b_value);
          return {CasSimplify(Product::Make({a_new, b_new}), var_intervals)};
        }
        return {_a, _b};
      };

      if (ap && ap->is_constant() && !bp && b.is_constant()) {
        return one_is_power(a, b);
      } else if (!ap && a.is_constant() && bp && bp->is_constant()) {
        return one_is_power(b, a);
      }
    }

    if (operands.size() == 2) {  // as sum
      auto a      = CasSimplify(operands[0], var_intervals);
      auto b      = CasSimplify(operands[1], var_intervals);
      auto* a_sum = a.As<Sum>();
      auto* b_sum = b.As<Sum>();

      if (b_sum) {
        std::vector<Expr> args;
        for (auto& v : b_sum->operands()) {
          args.push_back(CasSimplify(Product::Make({a, v}), var_intervals));
        }
        return {SimplifySum(Sum::Make(args))};
      }

      if (a_sum) {
        std::vector<Expr> args;
        for (auto& v : a_sum->operands()) {
          args.push_back(CasSimplify(Product::Make({b, v}), var_intervals));
        }
        return {SimplifySum(Sum::Make(args))};
      }
    }

    // case 4, b <| a
    {
      if (ExprPosCmp()(b, a)) {
        return {b, a};
      }
    }

    // case 5
    return operands;
  }

  // SPRDREC-2, Page 101
  if (operands.size() == 2 && (operands[0].As<Product>() || operands[1].As<Product>())) {
    auto a = CasSimplify(operands[0], var_intervals);
    auto b = CasSimplify(operands[1], var_intervals);

    auto* a_product = a.As<Product>();
    auto* b_product = b.As<Product>();
    // case 1
    if (a_product && b_product) {
      return MergeProduct(a_product->operands(), b_product->operands());
    }

    // case 2
    if (a_product) {
      return MergeProduct(a_product->operands(), {b});
    }

    // case 3
    if (b_product) {
      return MergeProduct({a}, b_product->operands());
    }
  }

  // SPRDREC-3
  if (operands.size() > 2) {
    auto p0 = CasSimplify(operands[0], var_intervals);
    auto w  = SimplifyProductRec(Rest(operands));
    if (p0.As<Product>()) {
      return MergeProduct(p0->operands, w);
    } else {
      return MergeProduct({p0}, w);
    }
  }

  return operands;
}

Expr CasSimplifyMutator::SimplifyProduct(Expr a) {
  a = SumOrProductGetSingleElementsRec(a);
  // We reuse the Mul node for production.
  auto* prod = a.As<Product>();
  if (!prod) return a;

  const auto& _operands = prod->operands();
  std::vector<Expr> operands;
  for (auto& e : _operands) operands.push_back(CasSimplify(e, var_intervals));
#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& v : operands) {
      ss << v << " ";
    }
    VLOG(6) << "operands: " << ss.str();
  };
#endif

  // SPRD-2
  // 0*x... = 0
  for (auto& opr : operands) {
    auto* opri = opr.As<IntImm>();
    auto* oprf = opr.As<FloatImm>();
    if (opri && opri->value == 0) return make_const(a.type(), 0);
    if (oprf && oprf->value == 0) return make_const(a.type(), 0);
  }

  // SPRD-3
  // prod(x) = x, single number.
  if (operands.size() == 1) {
    auto* first_s = operands.front().As<Sum>();
    auto* first_p = operands.front().As<Product>();
    return operands[0];
  }

  // SPRD-4
  return Product::Make(SimplifyProductRec(operands));
}

Expr CasSimplifyMutator::SimplifySum(Expr u) {
  u = SumOrProductGetSingleElementsRec(u);

  auto* sum = u.As<Sum>();
  CHECK(sum);

  auto& operands = sum->operands();
  if (operands.size() == 1) {
    return SimplifySum(operands[0]);
  }
  auto args = SimplifySumRec(operands);
  if (args.empty()) return make_const(u.type(), 0);
  if (args.size() == 1) return args[0];
  return Sum::Make(args);
}

// This implementation is similar to MergeProduct
std::vector<Expr> CasSimplifyMutator::MergeSum(const std::vector<Expr>& _p, const std::vector<Expr>& _q) {
  std::vector<Expr> p, q;
  for (auto& e : _p) {
    p.push_back(CasSimplify(e, var_intervals));
  }
  for (auto& e : _q) {
    q.push_back(CasSimplify(e, var_intervals));
  }

#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& x : p) ss << x << " ";

    VLOG(6) << "MergeSum p(" << ss.str() << ")";
    ss.str("");

    for (auto& x : q) ss << x << " ";
    VLOG(6) << "MergeSum q(" << ss.str() << ")";
    ss.str("");
  }
#endif
  // MPRD-1,2
  if (p.empty()) return q;
  if (q.empty()) return p;

  // MPRD-3
  auto p1 = p[0];
  auto q1 = q[0];
  auto h  = SimplifySumRec({p1, q1});

  // case 1
  if (h.empty()) {
    return MergeSum(Rest(p), Rest(q));
  }

  // case 2
  if (h.size() == 1) {
    auto rest = MergeSum(Rest(p), Rest(q));
    if (h[0].is_constant() && h[0].get_constant() == 0) return rest;
    rest.insert(std::begin(rest), h[0]);
    return rest;
  }

  // case 3
  if (h.size() == 2 && h[0] == p1 && h[1] == q1) {
    auto rest = MergeSum(Rest(p), q);
    rest.insert(std::begin(rest), p1);
    return rest;
  }

  // case 4
  if (h.size() == 2 && h[0] == q1 && h[1] == p1) {
    auto rest = MergeSum(p, Rest(q));
    rest.insert(std::begin(rest), q1);
    return rest;
  }

  // rest
  return Concat(p, q);
}

// The implementation is similar to SimpifyProductRec
std::vector<Expr> CasSimplifyMutator::SimplifySumRec(const std::vector<Expr>& _operands) {
  CHECK_GE(_operands.size(), 2UL);

  std::vector<Expr> operands;
  for (auto& e : _operands) operands.push_back(CasSimplify(e, var_intervals));

#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& o : operands) {
      ss << o.node_type() << " " << o << " ";
    }
    VLOG(6) << "SimplifySumRec operands: " << ss.str();
  }
#endif

  // SPRDREC-1
  if (operands.size() == 2 && !operands[0].As<Sum>() && !operands[1].As<Sum>()) {
    auto a = operands[0];
    auto b = operands[1];

    auto* ai = a.As<IntImm>();
    auto* af = a.As<FloatImm>();
    auto* bi = b.As<IntImm>();
    auto* bf = b.As<FloatImm>();

    // case 1, both are constants
    if (a.is_constant() && b.is_constant()) {
      if (ai) return {make_const(a.type(), ai->value + bi->value)};
      if (af) return {make_const(a.type(), af->value + bf->value)};
    }

    // cinn_min/cinn_max(a, b)+c = cinn_min/cinn_max(a+c, b+c)
    // c + cinn_min/cinn_max(a, b) = cinn_min/cinn_max(a+c, b+c)
    auto* a_min = a.As<Min>();
    auto* a_max = a.As<Max>();
    auto* b_min = b.As<Min>();
    auto* b_max = b.As<Max>();
    if (a_min) {
      return {CasSimplify(Min::Make(CasSimplify(Sum::Make({a_min->a(), b}), var_intervals),
                                    CasSimplify(Sum::Make({a_min->b(), b}), var_intervals)),
                          var_intervals)};
    }
    if (a_max) {
      return {CasSimplify(Max::Make(CasSimplify(Sum::Make({a_max->a(), b}), var_intervals),
                                    CasSimplify(Sum::Make({a_max->b(), b}), var_intervals)),
                          var_intervals)};
    }
    if (b_min) {
      return {CasSimplify(Min::Make(CasSimplify(Sum::Make({b_min->a(), a}), var_intervals),
                                    CasSimplify(Sum::Make({b_min->b(), a}), var_intervals)),
                          var_intervals)};
    }
    if (b_max) {
      return {CasSimplify(Max::Make(CasSimplify(Sum::Make({b_max->a(), a}), var_intervals),
                                    CasSimplify(Sum::Make({b_max->b(), a}), var_intervals)),
                          var_intervals)};
    }

    // case 2
    // x*1 -> a
    if (ai && ai->value == 0) return {b};
    if (af && af->value == 0.f) return {b};
    // 1*x -> x
    if (bi && bi->value == 0) return {a};
    if (bf && bf->value == 0.f) return {a};

    // customied case for Mod
    {
      auto* am = a.As<Mod>();
      auto* bm = b.As<Mod>();
      if (am && bm) {
        if (am->b() == bm->b() && ProductGetNonConstantPart(am->a()) == ProductGetNonConstantPart(bm->a())) {
          return {CasSimplify(Mod::Make(Sum::Make({am->a(), bm->a()}), am->b()), var_intervals)};
        }
      }
    }

    // case 3
    // Here is different from SimplifySumRec, to deal with cases like 3x + (-2x) = 2x
    auto a_non_constant = ProductGetNonConstantPart(a);
    auto b_non_constant = ProductGetNonConstantPart(b);
    if (a_non_constant.defined() && b_non_constant.defined() && a_non_constant == b_non_constant) {
      VLOG(3) << "a " << a;
      VLOG(3) << "b " << b;
      Expr s = SimplifySum(Sum::Make({ProductGetConstantPart(a), ProductGetConstantPart(b)}));
      Expr p = Product::Make({s, ProductGetNonConstantPart(a)});
      return {CasSimplify(p, var_intervals)};
    }

    // case 4, b <| a
    {
      if (ExprPosCmp()(b, a)) {
        return {b, a};
      }
    }

    // case 5
    return operands;
  }

  // SPRDREC-2, Page 101
  if (operands.size() == 2 && (operands[0].As<Sum>() || operands[1].As<Sum>())) {
    auto a = operands[0];
    auto b = operands[1];

    auto* a_sum = a.As<Sum>();
    auto* b_sum = b.As<Sum>();

    // case 1
    if (a_sum && b_sum) {
      return MergeSum(a_sum->operands(), b_sum->operands());
    }

    // case 2
    if (a_sum) {
      return MergeSum(a_sum->operands(), {b});
    }

    // case 3
    if (b_sum) {
      return MergeSum({a}, b_sum->operands());
    }
  }

  // SPRDREC-3
  if (operands.size() > 2) {
    auto p0 = operands[0];
    auto w  = SimplifySumRec(Rest(operands));
    if (p0.As<Sum>()) {
      return MergeSum(p0->operands, w);
    } else {
      return MergeSum({p0}, w);
    }
  }

  return operands;
}

void CasSimplifyMutator::AddBaseAndSimplify(Expr* base, Expr bound) {
  if ((*base).defined()) {
    *base = Sum::Make({*base, bound});
  } else {
    *base = bound;
  }
  *base = CasSimplify(*base, var_intervals);
}

void CasSimplifyMutator::UnfoldBound(Expr* lower_bound, Expr* upper_bound, Expr var, bool unfold_const_bound) {
  CHECK(lower_bound);
  CHECK(upper_bound);
  auto v_var = var.As<_Var_>();
  CHECK(v_var);
  if (var_intervals.count(v_var->name)) {
    auto& interval = var_intervals.at(v_var->name);
    if (interval.e_l.defined() && interval.e_r.defined()) {
      AddBaseAndSimplify(lower_bound, interval.e_l);
      AddBaseAndSimplify(upper_bound, interval.e_r);
    } else if (unfold_const_bound) {
      // unfold var's const bound
      AddBaseAndSimplify(lower_bound, Expr(interval.l));
      AddBaseAndSimplify(upper_bound, Expr(interval.r));
    } else {
      // no unfold var's const bound for var simplification
      AddBaseAndSimplify(lower_bound, var);
      AddBaseAndSimplify(upper_bound, var);
    }
  } else if (!unfold_const_bound) {
    // not get var's bound for var simplification
    AddBaseAndSimplify(lower_bound, var);
    AddBaseAndSimplify(upper_bound, var);
  } else {
    LOG(FATAL) << "can't get the bound";
  }
}

bool CasSimplifyMutator::GetVarBound(Expr* lower_bound, Expr* upper_bound, Expr var, bool unfold_const_bound) {
  CHECK(lower_bound);
  CHECK(upper_bound);
  auto v_var     = var.As<_Var_>();
  auto v_product = var.As<Product>();
  if (v_var && (var_intervals.count(v_var->name) || !unfold_const_bound)) {
    UnfoldBound(lower_bound, upper_bound, var, unfold_const_bound);
    return true;
  } else if (v_product) {
    // only deal with 2*x
    Expr p_lower_bound;
    Expr p_upper_bound;
    Expr const_oper     = ProductGetConstantPart(var);
    Expr non_const_oper = ProductGetNonConstantPart(var);
    auto v_var          = non_const_oper.As<_Var_>();
    if (v_var && var_intervals.count(v_var->name)) {
      Expr v_lower, v_upper;
      UnfoldBound(&v_lower, &v_upper, non_const_oper, unfold_const_bound);
      auto const_v = const_oper.get_constant();
      CHECK(v_lower.defined() && v_upper.defined());
      if (const_v > 0) {
        p_lower_bound = Product::Make({const_oper, v_lower});
        p_upper_bound = Product::Make({const_oper, v_upper});
      } else {
        p_lower_bound = Product::Make({const_oper, v_upper});
        p_upper_bound = Product::Make({const_oper, v_lower});
      }
      AddBaseAndSimplify(lower_bound, p_lower_bound);
      AddBaseAndSimplify(upper_bound, p_upper_bound);
      return true;
    }
  }
  return false;
}

bool CasSimplifyMutator::GetOperandBound(Expr* lower_bound, Expr* upper_bound, Expr v, bool unfold_const_bound) {
  // only support simple operand of int, var and var's product with int
  CHECK(lower_bound);
  CHECK(upper_bound);
  auto* v_int = v.As<IntImm>();
  if (v_int) {
    AddBaseAndSimplify(lower_bound, v);
    AddBaseAndSimplify(upper_bound, v);
    return true;
  } else if (GetVarBound(lower_bound, upper_bound, v, unfold_const_bound)) {
    return true;
  }
  return false;
}

bool CasSimplifyMutator::GetSumBound(Expr* lower_bound, Expr* upper_bound, Expr sum, bool unfold_const_bound) {
  // only support sum of int, var and var's product with int
  CHECK(lower_bound);
  CHECK(upper_bound);
  auto bound_sum = sum.As<Sum>();
  // CHECK(bound_sum);
  bool get_bound = true;
  Expr sum_lower_bound, sum_upper_bound;
  if (bound_sum) {
    for (Expr& v : bound_sum->operands()) {
      if (!GetOperandBound(&sum_lower_bound, &sum_upper_bound, v, unfold_const_bound)) {
        get_bound = false;
        break;
      }
    }
    if (get_bound) {
      *lower_bound = sum_lower_bound;
      *upper_bound = sum_upper_bound;
    }
    return get_bound;
  }
  return false;
}

bool CasSimplifyMutator::GetExprBound(Expr* lower_bound, Expr* upper_bound, Expr expr, bool unfold_const_bound) {
  // only support min's operands as sum, int or var or var's product with int or min/max
  auto bound_sum = expr.As<Sum>();
  auto bound_min = expr.As<Min>();
  auto bound_max = expr.As<Max>();
  bool get_bound = true;
  if (bound_sum) {
    get_bound = GetSumBound(lower_bound, upper_bound, expr, unfold_const_bound);
  } else if (bound_min) {
    get_bound = GetMinBound(lower_bound, upper_bound, expr, unfold_const_bound);
  } else if (bound_max) {
    get_bound = GetMaxBound(lower_bound, upper_bound, expr, unfold_const_bound);
  } else if (!GetOperandBound(lower_bound, upper_bound, expr, unfold_const_bound)) {
    return false;
  }
  return get_bound;
}

bool CasSimplifyMutator::GetMinBound(Expr* lower_bound, Expr* upper_bound, Expr min, bool unfold_const_bound) {
  // only support min's operands as sum, int or var or var's product with int or min/max
  auto bound_min = min.As<Min>();
  CHECK(bound_min);
  bool get_bound = true;
  Expr a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound;
  get_bound = get_bound && GetExprBound(&a_lower_bound, &a_upper_bound, bound_min->a(), unfold_const_bound) &&
              GetExprBound(&b_lower_bound, &b_upper_bound, bound_min->b(), unfold_const_bound);
  if (get_bound) {
    *lower_bound = CasSimplify(Min::Make(a_lower_bound, b_lower_bound), var_intervals);
    *upper_bound = CasSimplify(Min::Make(a_upper_bound, b_upper_bound), var_intervals);
  }
  return get_bound;
}

bool CasSimplifyMutator::GetMaxBound(Expr* lower_bound, Expr* upper_bound, Expr max, bool unfold_const_bound) {
  auto bound_max = max.As<Max>();
  CHECK(bound_max);
  bool get_bound = true;
  Expr a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound;
  get_bound = get_bound && GetExprBound(&a_lower_bound, &a_upper_bound, bound_max->a(), unfold_const_bound) &&
              GetExprBound(&b_lower_bound, &b_upper_bound, bound_max->b(), unfold_const_bound);
  if (get_bound) {
    *lower_bound = CasSimplify(Max::Make(a_lower_bound, b_lower_bound), var_intervals);
    *upper_bound = CasSimplify(Max::Make(a_upper_bound, b_upper_bound), var_intervals);
  }
  return get_bound;
}

bool CasSimplifyMutator::SimplifySpecificSumMod(Expr* result, Expr a, Expr b) {
// case1: (32+(-x))%33 = 32-x%33 (0<=x<=32)
// case2: (x-32))%33 = x%33 - 32%33 (0<=x<=32)
#ifdef CINN_WITH_CUDA
  return false;
#else
  auto a_sum = a.As<Sum>();
  auto b_i   = b.As<IntImm>();
  if (!a_sum || !b_i) {
    return false;
  }
  int const_value = 0;
  Expr lower_bound;
  Expr upper_bound;
  Expr rest_oper;
  bool can_simplify = true;
  bool has_int      = false;
  // fold only the expr bound(may contains the var) and try to simplify the var
  Expr unfolded_lower_bound, unfolded_upper_bound;
  for (Expr& v : a_sum->operands()) {
    auto* v_int     = v.As<IntImm>();
    auto* v_var     = v.As<_Var_>();
    auto* v_product = v.As<Product>();
    if (v_int) {
      const_value += v_int->value;
      has_int = true;
    } else if (GetVarBound(&lower_bound, &upper_bound, v, false)) {
      AddBaseAndSimplify(&rest_oper, v);
    } else {
      can_simplify = false;
      break;
    }
  }
  can_simplify = can_simplify && has_int && std::abs(const_value) % b_i->value == b_i->value - 1 &&
                 lower_bound.defined() && upper_bound.defined() && rest_oper.defined();
  // further infer the vars' bound by the intervals infos, try to get the constant
  if (can_simplify) {
    std::vector<Expr> bounds = {lower_bound, upper_bound};
    for (int i = 0; i < bounds.size(); ++i) {
      Expr bound      = bounds[i];
      auto* bound_sum = bound.As<Sum>();
      auto* bound_min = bound.As<Min>();
      auto* bound_max = bound.As<Max>();
      Expr bound_l, bound_r;
      GetExprBound(&bound_l, &bound_r, bound);
      if (i == 0 && bound_l.defined()) {
        lower_bound = bound_l;
      }
      if (i == 1 && bound_r.defined()) {
        upper_bound = bound_r;
      }
    }
  } else {
    return false;
  }
  // case1: (32+(-x))%33 = 32-x%33 (0<=x<=32)
  // case2: (x-32)%33 = x%33 - 32%33 (0<=x<=32)
  can_simplify = can_simplify && lower_bound.is_constant();
  bool case1   = can_simplify && const_value >= 0 && lower_bound.get_constant() >= -const_value &&
               upper_bound.is_constant() && upper_bound.get_constant() <= 0;
  bool case2 = can_simplify && const_value <= 0 && lower_bound.get_constant() >= 0 && upper_bound.is_constant() &&
               upper_bound.get_constant() <= -const_value;
  can_simplify = can_simplify && (case1 || case2);
  if (can_simplify) {
    Expr const_expr;
    if (const_value < 0) {
      const_expr = make_const(b->type(), const_value % b_i->value);
    } else {
      const_expr = make_const(b->type(), const_value % b_i->value);
    }
    *result = CasSimplify(Sum::Make({const_expr, CasSimplify(Mod::Make(rest_oper, b), var_intervals)}), var_intervals);
    return true;
  }
  return false;
#endif
}

Expr CasSimplifyMutator::SimplifyMod(Expr u) {
  auto* node = u.As<Mod>();
  CHECK(node);

  auto a = CasSimplify(node->a(), var_intervals);
  auto b = CasSimplify(node->b(), var_intervals);

  auto* a_i       = a.As<IntImm>();
  auto* a_product = a.As<Product>();
  auto* a_sum     = a.As<Sum>();
  auto* a_var     = a.As<_Var_>();
  auto* a_mod     = a.As<Mod>();

  auto* b_i = b.As<IntImm>();

  // 7 % 3
  if (a_i && b_i) {
    return make_const(a_i->type(), a_i->value % b_i->value);
  }

  // x % 1 = 0
  if (b_i && b_i->value == 1) return make_const(b_i->type(), 0);

  // 2x % 2 = 0
  if (b_i && a_product && a_product->operand(0).As<IntImm>()) {
    if (a_product->operand(0).As<IntImm>()->value % b_i->value == 0) return make_const(a_product->type(), 0);
  }

  // 0 % x = 1, 1 % x = 1
  if (a_i && (a_i->value == 0 || a_i->value == 1)) return a;

  if (b_i && a_var && var_intervals.count(a_var->name)) {
    auto& interval = var_intervals.at(a_var->name);
    int b_abs      = std::abs(b_i->value);
    // x\in[1, 3] % 4 = x
    if (std::abs(interval.l) < b_abs && std::abs(interval.r) < b_abs) return a;
    // [3,3] % 3 = 0
    if (interval.l == interval.r && interval.l % b_abs == 0) return make_const(b_i->type(), 0);
  }

  if (a_product && b_i) {
    if (IsDivisible(a_product, b_i->value)) {
      return make_const(Int(32), 0);
    }
  }

  // (2x+y+z) % 2 = (y+z) % 2
  if (a_sum && b_i) {
    std::vector<Expr> sum_args;
    for (auto& v : a_sum->operands()) {
      if (!IsDivisible(v, b_i->value)) {
        sum_args.push_back(v);
      }
    }

    if (sum_args.empty()) return make_const(b_i->type(), 0);
    if (sum_args.size() == 1) {
      return SimplifyMod(Mod::Make(sum_args.front(), b));
    }
    if (sum_args.size() == a_sum->operands().size()) {
      if (b_i->value > 0 && !var_intervals.empty()) {
        // case1: (32+(-x))%33 = 32-x%33 (0<=x<=32)
        // case2: (x-32))%33 = x%33 - 32%33 (0<=x<=32)
        Expr result;
        if (SimplifySpecificSumMod(&result, a, b)) {
          return result;
        }
      }

      return Mod::Make(a, b);
    }
    return SimplifyMod(Mod::Make(Sum::Make(sum_args), b));
  }

  return Mod::Make(a, b);
}

Expr CasSimplifyMutator::SimplifyMinAndMax(Expr u) {
  // simplify min/max
  auto* u_max = u.As<Max>();
  auto* u_min = u.As<Min>();
  if (u_max) {
    Expr a = CasSimplify(u_max->a(), var_intervals);
    Expr b = CasSimplify(u_max->b(), var_intervals);
    if (a.is_constant() && b.is_constant()) {
      return a.get_constant() >= b.get_constant() ? a : b;
    }
    return ir::Max::Make(a, b);
  }

  if (u_min) {
    Expr a = CasSimplify(u_min->a(), var_intervals);
    Expr b = CasSimplify(u_min->b(), var_intervals);
    if (a.is_constant() && b.is_constant()) {
      return a.get_constant() <= b.get_constant() ? a : b;
    }
    return ir::Min::Make(a, b);
  }
  return u;
}

Expr CasSimplifyMutator::SimplifyCmp(Expr u) {
  Expr a = operator()(u->operand(0));
  Expr b = operator()(u->operand(1));

  if (a.is_constant() && b.is_constant()) {
    switch (u->node_type()) {
      case ir::IrNodeTy::LT:
        return Expr(a.get_constant() < b.get_constant());
      case ir::IrNodeTy::LE:
        return Expr(a.get_constant() <= b.get_constant());
      case ir::IrNodeTy::GT:
        return Expr(a.get_constant() > b.get_constant());
      case ir::IrNodeTy::GE:
        return Expr(a.get_constant() >= b.get_constant());
      case ir::IrNodeTy::EQ:
        return Expr(a.get_constant() == b.get_constant());
      case ir::IrNodeTy::NE:
        return Expr(a.get_constant() != b.get_constant());
    }
  }

  return u;
}

/**
 * deal with index's div-mod add simplification, tempory solution, not cover all situations.
 * case 1: m / n * n + m % n = m (m, n's type is int)
 * case 2: m / n1 * n3 + n2 * m % n3 = n2 * m if n3 = n1 * n2 (m, n1, n2, n3's type is int)
 * case 3: m / n2 + n1 * m % (n3) = n1 * m if n3 = n1 * n2 (m, n1, n2, n3's type is int)
 */
Expr CasSimplifyMutator::SimplifySpecificSum(Expr tmp) {
  auto sum = tmp.As<Sum>();
  if (!sum) {
    return tmp;
  }
  CHECK_GE(sum->operands().size(), 2U);
  Expr left      = sum->operand(0);
  Expr right     = sum->operand(1);
  auto left_mod  = left.As<Mod>();
  auto right_mod = right.As<Mod>();
  auto left_mul  = left.As<Product>();
  auto right_mul = right.As<Product>();
  auto left_div  = left.As<FracOp>();
  auto right_div = right.As<FracOp>();
  // normalize to left mul and right mod
  if (right_mul && left_mod) {
    left_mul  = right_mul;
    right_mod = left_mod;
  }
  // normalize to left div and right mod
  if (right_div && left_mod) {
    left_div  = right_div;
    right_mod = left_mod;
  }
  if (!right_mod || (!left_mul && !left_div)) {
    return tmp;
  }
  CHECK_GE(right_mod->operands().size(), 2U);
  Expr mod_left  = right_mod->operand(0);
  Expr mod_right = right_mod->operand(1);
  if (!mod_left->type().is_integer() || !mod_right->type().is_integer()) {
    return tmp;
  }
  if (left_mul) {
    // case 1: m / n * n + m % n = m (m, n's type is int)
    // case 2: m / n1 * n3 + n2 * m % n3 = n2 * m if n3 = n1 * n2 (m, n1, n2, n3's type is int)
    CHECK_GE(left_mul->operands().size(), 2U);
    Expr mul_left  = left_mul->operand(0);
    Expr mul_right = left_mul->operand(1);
    if (!MathEqual(mod_right, mul_right)) {
      return tmp;
    }
    auto div = mul_left.As<FracOp>();
    if (!div) {
      return tmp;
    }
    CHECK_GE(div->operands().size(), 2U);
    Expr div_left  = div->operand(0);
    Expr div_right = div->operand(1);
    if (!div_left->type().is_integer() || !div_right->type().is_integer()) {
      return tmp;
    }
    if (MathEqual(div_left * mod_right, mod_left * div_right)) {
      tmp = mod_left;
    }
  } else if (left_div) {
    // case 3: m / n1 + n2 * m % (n3) = n2 * m if n3 = n1 * n2 (m, n1, n2, n3's type is int)
    CHECK_GE(left_div->operands().size(), 2U);
    Expr div_left  = left_div->operand(0);
    Expr div_right = left_div->operand(1);
    if (!div_left->type().is_integer() || !div_right->type().is_integer()) {
      return tmp;
    }
    if (MathEqual(mod_right * div_left, mod_left * div_right)) {
      tmp = mod_left;
    }
  }

  return tmp;
}

Expr CasSimplifyMutator::operator()(Expr u) {
  if (u.As<Min>() || u.As<Max>()) {
    return SimplifyMinAndMax(u);
  }

  u = detail::SumOrProductGetSingleElementsRec(u);

  if (u.is_constant() || u.As<_Var_>()) return u;

  if (u.As<Power>()) {
    auto expr = SimplifyPower(u);
    return expr;
  }

  if (u.As<FracOp>()) {
    u        = SimplifyFracOp(u);
    auto tmp = FurtherSimplifyFracWithInterval(u, var_intervals);
    if (!tmp.same_as(u)) return operator()(tmp);
    return u;
  }

  if (u.As<Product>()) {
    return detail::SumOrProductGetSingleElementsRec(SimplifyProduct(u));
  }

  if (u.As<Sum>()) {
    auto tmp = detail::SumOrProductGetSingleElementsRec(SimplifySum(u));
    // deal with index's div-mod add simplification, tempory solution, not cover all situations.
    // case 1: m / n * n + m % n = m (m, n's type is int)
    // case 2: m / n1 * n3 + n2 * m % n3 = n2 * m if n3 = n1 * n2 (m, n1, n2, n3's type is int)
    // case 3: m / n2 + n1 * m % (n3) = n1 * m if n3 = n1 * n2 (m, n1, n2, n3's type is int)
    return SimplifySpecificSum(tmp);
  }

  if (u.As<Mod>()) {
    return detail::SumOrProductGetSingleElementsRec(SimplifyMod(u));
  }

  if (u.is_cmp()) {
    return SimplifyCmp(u);
  }

  switch (u.node_type()) {
    case ir::IrNodeTy::And:
    case ir::IrNodeTy::Or:
    case ir::IrNodeTy::Not:
      return SimplifyCond(u);
    default:
      break;
  }

  return u;
}

bool CASasSymbol(Expr expr) {
  auto* load_n      = expr.As<Load>();
  auto* var_n       = expr.As<_Var_>();
  auto* broadcast_n = expr.As<Broadcast>();

  return load_n || var_n || broadcast_n;
}

Expr ConvertCinnToCAS(Expr expr) {
  Expr copied = optim::IRCopy(expr);

  struct Mutator : public ir::IRMutator<ir::Expr*> {
    void operator()(Expr* expr) { Visit(expr); }
    void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const Add* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      bool is_zero_a = a.is_constant() && a.get_constant() == 0;
      bool is_zero_b = b.is_constant() && b.get_constant() == 0;
      if (is_zero_a) {
        *expr = b;
        return;
      } else if (is_zero_b) {
        *expr = a;
        return;
      }
      *expr = Sum::Make({a, b});
    }
    void Visit(const Mul* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      if (a.is_constant() && a.get_constant() == 0) {
        *expr = make_const(a->type(), 0);
        return;
      }

      if (b.is_constant() && b.get_constant() == 0) {
        *expr = make_const(b->type(), 0);
        return;
      }

      *expr = Product::Make({a, b});
    }

    void Visit(const Sub* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      bool is_zero_a = a.is_constant() && a.get_constant() == 0;
      bool is_zero_b = b.is_constant() && b.get_constant() == 0;
      if (is_zero_a) {
        *expr = Product::Make({make_const(b->type(), -1), b});
        return;
      } else if (is_zero_b) {
        *expr = a;
        return;
      }

      b     = Product::Make({make_const(b->type(), -1), b});
      *expr = Sum::Make({a, b});
    }

    void Visit(const Div* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      CHECK(!is_zero(b)) << "Dividend should not be zero";

      if (a.is_constant() && a.get_constant() == 0) {
        *expr = make_const(a->type(), 0);
        return;
      }

      if (a.type().is_float()) {
        b     = Power::Make(b, make_const(Int(32), -1));
        *expr = Product::Make({a, b});
      } else {  // int division, NOTE that 3/2 = 1, 3./2 = 1.5
        *expr = FracOp::Make(a, b);
      }
    }

    void Visit(const Minus* op, Expr* expr) override {
      auto a = op->v();

      Visit(&a);

      if (a.is_constant()) {
        auto value = a.get_constant();
        if (value == 0) {
          *expr = make_const(a->type(), 0);
          return;
        }
      }

      *expr = Product::Make({make_const(a->type(), -1), a});
    }
  };

  Mutator()(&copied);
  return copied;
}

Expr ConvertCasToCinn(Expr expr) {
  Expr copied = optim::IRCopy(expr);

  struct Mutator : ir::IRMutator<Expr*> {
    void operator()(Expr* expr) { Visit(expr); }
    void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const Product* op, Expr* expr) override {
      std::vector<Expr> operands;
      auto* node = expr->As<Product>();
      for (auto& v : node->operands()) {
        auto c = v;
        Mutator()(&c);
        operands.push_back(c);
      }

      CHECK(!operands.empty());
      if (operands.size() == 1) {
        *expr = operands[0];
      } else if (operands.size() == 2) {
        *expr = Mul::Make(operands[0], operands[1]);
      } else {
        auto a = operands[0];
        auto b = Product::Make(Rest(operands));
        Mutator()(&b);
        *expr = Mul::Make(a, b);
      }

      // process the Mul
      Visit(expr);
    }

    void Visit(const Sum* op, Expr* expr) override {
      std::vector<Expr> operands;
      auto* node = expr->As<Sum>();
      for (auto& v : node->operands()) {
        auto c = v;
        Mutator()(&c);
        operands.push_back(c);
      }

      CHECK(!operands.empty());
      if (operands.size() == 1) {
        *expr = operands[0];
      } else if (operands.size() == 2) {
        *expr = Add::Make(operands[0], operands[1]);
      } else {
        auto a = operands[0];
        auto b = Sum::Make(Rest(operands));
        Mutator()(&b);
        *expr = Add::Make(a, b);
      }

      // process the sum
      Visit(expr);
    }

    void Visit(const FracOp* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      CHECK(!is_zero(b)) << "Dividend should not be zero";
      *expr = Div::Make(a, b);
      Visit(expr);
    }

    // a * b^-1 -> a/b
    void Visit(const Mul* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      auto* bp = b.As<ir::Power>();
      if (bp && bp->b().is_constant() && bp->b().get_constant() == -1.f) {
        *expr = Div::Make(a, bp->a());
      } else {
        *expr = Mul::Make(a, b);
      }
    }

    // a + -1*b -> a-b
    void Visit(const Add* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      auto* bp = b.As<ir::Mul>();
      if (bp && bp->a().is_constant() && bp->a().get_constant() == -1.f) {
        *expr = Sub::Make(a, bp->b());
      } else {
        *expr = Add::Make(a, b);
      }
    }

    void Visit(const Power* op, Expr* expr) override {
      auto a = op->a();
      auto b = op->b();

      Visit(&a);
      Visit(&b);

      auto* node = expr->As<ir::Power>();
      if (b.is_constant()) {
        if (b.get_constant() == 1) {
          *expr = a;
        } else if (b.get_constant() == 0) {
          *expr = make_const(a.type(), 1);
        } else if (b.get_constant() > 0) {
          auto init = a;
          for (int i = 0; i < b.get_constant() - 1; i++) {
            init = init * a;
          }
          *expr = init;
        } else {
          // some case like a^-2
          auto new_expr = make_const(a.type(), 1.f) / (ir::Power::Make(a, make_const(b.type(), -b.get_constant())));
          Visit(&new_expr);
          *expr = new_expr;
          return;
        }
      } else {
        CINN_NOT_IMPLEMENTED
      }
    }
  };

  Mutator()(&copied);
  return copied;
}

bool IsExprCasCompatible(Expr expr) {
  auto teller = [](const Expr* expr) {
    return expr->As<Add>() || expr->As<Sub>() || expr->As<Mul>() || expr->As<Div>();
  };
  return ir::CollectIRNodes(expr, teller).empty();
}

// Partially divide a by b. e.g. (2x+y)/2 => x + y/2
Expr DividePartially(Sum* a, int b) {
  std::vector<Expr> external_sum_args, sum_args;

  for (auto& item : a->operands()) {
    if (item.As<Product>() && IsDivisible(item.As<Product>(), b)) {
      external_sum_args.push_back(Divide(item.As<Product>(), b));
    } else if (item.As<IntImm>() && IsDivisible(item.As<IntImm>()->value, b)) {
      external_sum_args.push_back(make_const(item.type(), item.As<IntImm>()->value / b));
    } else {
      sum_args.push_back(item);
    }
  }

  if (!external_sum_args.empty()) {
    if (sum_args.empty()) return Sum::Make(external_sum_args);
    Expr internal_sum = sum_args.size() == 1 ? sum_args[0] : Sum::Make(sum_args);
    Expr new_frac     = FracOp::Make(internal_sum, make_const(a->type(), b));
    return Sum::Make(Concat(external_sum_args, {new_frac}));
  }
  return Expr(a);
}

bool IsMonotonical(Expr u, Var v) {
  auto* up = u.As<Product>();
  auto* uv = u.As<_Var_>();

  if (uv && uv->name == v->name) return true;
  if (up) {
    for (auto& item : up->operands()) {
      if (IsMonotonical(item, v)) return true;
    }
  }
  return false;
}

// Should be called after SimplifyFracOp. If y is integer and $y\in \[0, 3\]$, then y/4=0
Expr CasSimplifyMutator::FurtherSimplifyFracWithInterval(
    Expr expr, const std::unordered_map<std::string, CasInterval>& var_intervals) {
  auto* node = expr.As<FracOp>();
  if (!node) return expr;
  auto a = CasSimplify(node->a(), var_intervals);
  auto b = CasSimplify(node->b(), var_intervals);

  auto* ai = a.As<IntImm>();
  auto* bi = b.As<IntImm>();
  auto* av = a.As<_Var_>();
  auto* bv = b.As<_Var_>();
  auto* ap = a.As<Product>();
  // case: y / 4, y\in[0,3]
  if (bi) {
    if (av) {
      auto it = var_intervals.find(av->name);
      if (it != var_intervals.end() && std::abs(it->second.r) < std::abs(bi->value) &&
          std::abs(it->second.l) < std::abs(bi->value))
        return make_const(a.type(), 0);
    }
  }
  // case: 1/y, y\in(2, 100)
  if (ai) {
    if (bv) {
      auto it     = var_intervals.find(bv->name);
      auto ai_abs = std::abs(ai->value);
      if (it != var_intervals.end()) {
        LOG(INFO) << "found " << bv->name << " " << it->second << " "
                  << " ai " << ai_abs;
      }
      if (it != var_intervals.end() && std::abs(it->second.r) > ai_abs && std::abs(it->second.l) > ai_abs) {
        return make_const(a.type(), 0);
      }
    }
  }
  return expr;
}

Expr SimplifyConstantFrac(FracOp* node) {
  auto* ai = node->a().As<ir::IntImm>();
  auto* au = node->a().As<ir::UIntImm>();
  auto* af = node->a().As<ir::FloatImm>();

  if (ai) {
    auto* bi = node->b().As<ir::IntImm>();
    CHECK(bi);
    return make_const(ai->type(), ai->value / bi->value);
  }

  if (au) {
    auto* bu = node->b().As<ir::UIntImm>();
    CHECK(bu);
    return make_const(au->type(), au->value / bu->value);
  }

  if (af) {
    auto* bf = node->b().As<ir::FloatImm>();
    CHECK(af);
    return make_const(af->type(), af->value / bf->value);
  }
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr CasSimplifyMutator::SimplifyFracOp(Expr expr) {
  VLOG(3) << "CAS simplify Frac " << expr;
  auto* node = expr.As<FracOp>();
  auto a     = CasSimplify(node->a(), var_intervals);
  auto b     = CasSimplify(node->b(), var_intervals);

  auto* ap = a.As<Product>();
  auto* bp = b.As<Product>();
  auto* as = a.As<Sum>();
  auto* bi = b.As<IntImm>();
  auto* ai = a.As<IntImm>();
  auto* af = a.As<FloatImm>();
  auto* bf = b.As<FloatImm>();
  auto* av = a.As<_Var_>();
  auto* bv = b.As<_Var_>();

  // case 1
  // integer constant division: 64/3
  if (node->is_constant()) {
    if (int_compute_) {
      return SimplifyConstantFrac(node);
    } else {
      return SimplifyRationalNumber(expr);
    }
  }

  // case 2
  // sum/x or product/x is divisiable
  if (bi) {
    auto* a_sum     = a.As<Sum>();
    auto* a_product = a.As<Product>();
    // disiviable
    if (a_sum && IsDivisible(a_sum, bi->value)) return Divide(a_sum, bi->value);
    if (a_product) {
      if (IsDivisible(a_product, bi->value)) {
        return Divide(a_product, bi->value);
      } else {
        return FracOp::Make(a, b);
      }
    }
    // not divisiable
    /*
    if (a_sum) {
      auto expr = DividePartially(a_sum, bi->value);
      return expr;
    }
     */
  }

  // cinn_min/cinn_max(a, b)/2 = cinn_min/cinn_max(a/2, b/2)
  if ((bi && bi->value > 0) || (bf && bf->value > 0)) {
    auto cmp_min = a.As<Min>();
    auto cmp_max = a.As<Max>();
    if (cmp_min) {
      return {CasSimplify(Min::Make(CasSimplify(FracOp::Make(cmp_min->a(), b), var_intervals),
                                    CasSimplify(FracOp::Make(cmp_min->b(), b), var_intervals)),
                          var_intervals)};
    }
    if (cmp_max) {
      return {CasSimplify(Max::Make(CasSimplify(FracOp::Make(cmp_max->a(), b), var_intervals),
                                    CasSimplify(FracOp::Make(cmp_max->b(), b), var_intervals)),
                          var_intervals)};
    }
  }

  if (av && bi) {
    if (var_intervals.count(av->name)) {
      auto& interval = var_intervals.at(av->name);
      int b_abs      = std::abs(bi->value);
      if (std::abs(interval.l) < b_abs && std::abs(interval.r) < b_abs) return make_const(bi->type(), 0);
      return FracOp::Make(a, b);
    }
  }

  // (32x+y)/32 = x + y/32
  if (as && bi) {
    std::vector<Expr> external_sum_args;
    std::vector<Expr> internal_sum_args;
    for (auto& e : as->operands()) {
      if (IsDivisible(e, bi->value)) {
        if (e.As<Sum>()) external_sum_args.push_back(Divide(e.As<Sum>(), bi->value));
        if (e.As<IntImm>()) external_sum_args.push_back(make_const(bi->type(), e.As<IntImm>()->value / bi->value));
        if (e.As<Product>()) external_sum_args.push_back(Divide(e.As<Product>(), bi->value));
      } else {
        internal_sum_args.push_back(e);
      }
    }

    Expr external_sum, internal_sum;
    if (!external_sum_args.empty()) {
      if (external_sum_args.size() == 1)
        external_sum = external_sum_args.front();
      else
        external_sum = Sum::Make(external_sum_args);
    }

    if (!internal_sum_args.empty()) {
      internal_sum = FracOp::Make(Sum::Make(internal_sum_args), b);
    }

    if (external_sum.defined() && internal_sum.defined()) {
      return CasSimplify(Sum::Make({external_sum, internal_sum}), var_intervals);
    }
    if (external_sum.defined()) return CasSimplify(external_sum, var_intervals);
    return internal_sum;
  }

  // solve the case: 2abc / b
  // Both avs and bvs should be sorted first.
  auto reduce_product_div_product = [](const std::vector<Expr>& avs, const std::vector<Expr>& bvs) {
    std::vector<Expr> avs1, bvs1;
    int i = 0;
    int j = 0;

    ExprPosCmp cmp;

    while (i < avs.size() && j < bvs.size()) {
      auto& a = avs[i];
      auto& b = bvs[j];
      if (a.is_constant() && b.is_constant()) {
        auto* ai = a.As<IntImm>();
        auto* bi = b.As<IntImm>();
        auto* af = a.As<FloatImm>();
        auto* bf = b.As<FloatImm>();
        if (ai) {
          CHECK(bi);
          int g   = gcd(ai->value, bi->value);
          int a_d = ai->value / g;
          int b_d = bi->value / g;

          avs1.push_back(make_const(a.type(), a_d));
          if (b_d != 1) bvs1.push_back(make_const(b.type(), b_d));
        }

        CHECK(!af);
        i++;
        j++;
      } else if (avs[i] == bvs[j]) {
        i++;
        j++;
      } else {
        // <
        if (cmp(avs[i], bvs[j])) {
          avs1.push_back(avs[i++]);
        } else {
          bvs1.push_back(bvs[j++]);
        }
      }
    }
    while (i < avs.size()) {
      avs1.push_back(avs[i++]);
    }
    while (j < bvs.size()) {
      bvs1.push_back(bvs[j++]);
    }
    if (avs1.empty()) return make_const(avs[0].type(), 1);
    if (bvs1.empty()) return Product::Make(avs1);

    return FracOp::Make(Product::Make(avs1), Product::Make(bvs1));
  };

  {
    std::vector<Expr> a_args, b_args;
    if (ap)
      a_args = ap->operands();
    else
      a_args.push_back(a);
    if (bp)
      b_args = bp->operands();
    else
      b_args.push_back(b);

    return reduce_product_div_product(a_args, b_args);
  }

  // x / x
  if (a.type().is_int() && b.type().is_int() && av && bv) {
    if (a == b) return make_const(a.type(), 1);
  }

  if (node->a().same_as(a) && node->b().same_as(b)) return expr;
  return FracOp::Make(a, b);
}

Expr CasSimplifyMutator::SimplifyCond(Expr u) {
  switch (u->node_type()) {
      // -------------------------- NOT -----------------------------
    case ir::IrNodeTy::Not: {
      auto* node = u.As<ir::Not>();
      Expr v     = operator()(node->v());
      switch (v.node_type()) {
          // Not 1 = (1 == 0)
        case ir::IrNodeTy::IntImm:
          return Expr(v.As<IntImm>()->value == 0);
          // Not Not v = v
        case ir::IrNodeTy::Not:
          return v;
          // Not <= is >
        case ir::IrNodeTy::LE:
          return ir::GT::Make(v->operand(0), v->operand(1));
          // Not < is >=
        case ir::IrNodeTy::LT:
          return ir::GE::Make(v->operand(0), v->operand(1));
          // Not >= is <
        case ir::IrNodeTy::GE:
          return ir::LT::Make(v->operand(0), v->operand(1));
          // Not > is <=
        case ir::IrNodeTy::GT:
          return ir::LE::Make(v->operand(0), v->operand(1));
        default:
          return ir::Not::Make(v);
      }
    } break;
      // -------------------------- AND OR -----------------------------
    case ir::IrNodeTy::And:
    case ir::IrNodeTy::Or: {
      Expr a = operator()(u->operand(0));
      Expr b = operator()(u->operand(1));
      if (a.is_constant() || b.is_constant()) {
        if (u.As<ir::And>()) {
          // 1 && b is b
          if (a.As<ir::IntImm>()) {
            return a.As<ir::IntImm>()->value ? b : Expr(false);
          }
          // a && 1 is a
          if (b.As<ir::IntImm>()) {
            return b.As<ir::IntImm>()->value ? a : Expr(false);
          }
          return ir::And::Make(a, b);
        }
        if (u.As<ir::Or>()) {
          // 1 || b is 1
          if (a.As<ir::IntImm>()) {
            return a.As<ir::IntImm>()->value ? a : b;
          }
          // a || 1 is 1
          if (b.As<ir::IntImm>()) {
            return b.As<ir::IntImm>()->value ? b : a;
          }
        }
        return ir::Or::Make(a, b);
      }

      return u;
    }

    default:
      return u;
  }
}

}  // namespace detail

Expr CasSimplify(Expr u, const std::unordered_map<std::string, CasInterval>& var_intervals) {
  return detail::CasSimplifyMutator(var_intervals)(u);
}

Expr SolveInequality(Expr inequality, Var val) {
  auto copied = AutoSimplify(inequality);

  auto* le_n = copied.As<ir::LE>();
  auto* lt_n = copied.As<ir::LT>();
  auto* gt_n = copied.As<ir::GT>();
  auto* ge_n = copied.As<ir::GE>();

  Expr a, b;

#define __(x__)   \
  if (x__) {      \
    a = x__->a(); \
    b = x__->b(); \
  }
  __(le_n)
  __(lt_n)
  __(gt_n)
  __(ge_n)
#undef __

  Expr all = AutoSimplify(a - b);

  // if (common::IsPureMath(a) && common::IsPureMath(b)) {
  if (true) {
    auto [res, positive] = common::Solve(a, b, val);  // NOLINT
    // Simplify it with CAS to avoid random result from GiNac.
    res = AutoSimplify(res);
    res = common::cast(res, val->type());

    if (le_n) {
      if (positive) return ir::LE::Make(val, res);
      return ir::GE::Make(val, res);
    }
    if (lt_n) {
      if (positive) return ir::LT::Make(val, res);
      return ir::GT::Make(val, res);
    }
    if (ge_n) {
      if (positive) return ir::GE::Make(val, res);
      return ir::LE::Make(val, res);
    }
    if (gt_n) {
      if (positive) return ir::GT::Make(val, res);
      return ir::LT::Make(val, res);
    }
  } else {
    return AutoSimplify(inequality);
  }
  return Expr();
}

}  // namespace common
}  // namespace cinn

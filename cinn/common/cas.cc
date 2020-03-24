#include "cinn/common/cas.h"
#include <algorithm>
#include "cinn/common/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace common {
using namespace ir;  // NOLINT

int gcd(int a, int b) {
  // Everything divides 0
  if (a == 0) return b;
  if (b == 0) return a;

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
  return Expr(1);
}

// 3*x => x
// x => x
// x^-1 => x^-1
Expr ProductGetNonConstantPart(Expr u) {
  auto* product = u.As<Product>();
  if (product) {
    if (product->operand(0).is_constant()) {
      if (product->operands().size() == 2) return product->operands()[1];
      return Product::Make(Rest(product->operands()));
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

void SimplifyRNE(Expr* u) {
  NOT_IMPLEMENTED
  // get integer.
  if (u->As<IntImm>()) {
    return;
  }

  if (auto* frac_op = u->As<ir::Div>()) {
    if (frac_op->b().As<IntImm>() && frac_op->b().As<IntImm>()->value == 0) {
      LOG(FATAL) << "Get zero denominator";
    } else {
      return;
    }
  } else if ((*u)->operands.size() == 1) {
    SimplifyRNE(&(*u)->operands[0]);
    if (u->As<Add>()) return;
  } else if ((*u)->operands.size() == 2) {
    auto* add_n = u->As<Add>();
    auto* mul_n = u->As<Mul>();
    if (add_n || mul_n) {
      Expr& opr0 = add_n ? add_n->operand(0) : mul_n->operand(0);
      Expr& opr1 = add_n ? add_n->operand(1) : mul_n->operand(1);
      SimplifyRNE(&opr0);
      SimplifyRNE(&opr1);

      if (add_n) {
        detail::EvaluateSum(opr0, opr1);
        return;
      }
      if (mul_n) {
        detail::EvaluateSum(opr0, opr1);
        return;
      }
    }
  }
}

namespace detail {

inline int Iquot(int n, int d) { return n / d; }

inline int Irem(int n, int d) {
  int k = Iquot(n, d);
  return n - d * k;
}

Expr SimplifyRationalNumber(Expr u) {
  if (u.As<IntImm>()) return u;
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

Expr SimplifyIntegerPower(Expr u) {
  auto* node = u.As<Power>();
  CHECK(node);
  Expr v = node->a();
  Expr n = node->b();

  auto* ni = n.As<IntImm>();
  if (ni) {
    // x^0 = 1
    if (ni->value == 0) {
      return make_const(u.type(), 1);
    }
    if (ni->value == 1) {
      return v;
    }
  }

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

Expr SimplifyPower(Expr u) {
  auto* node = u.As<Power>();
  CHECK(node);
  Expr a = AutoSimplify(node->a());
  Expr b = AutoSimplify(node->b());

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
  auto * product = u.As<Product>();
  auto* sum = u.As<Sum>();
  if (product && product->operands().size() == 1) {
    return SumOrProductGetSingleElementsRec(u->operands.front());
  }
  if (sum && sum->operands().size() == 1) {
    return SumOrProductGetSingleElementsRec(u->operands.front());
  }
  return u;
}


// Order, reference to Page 85.
bool ExprPosCmp::operator()(const Expr& a, const Expr& b) {
  // O-1, 1 <| 2
  if (a.is_constant() && b.is_constant()) {
    return a.get_constant() < b.get_constant();
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

std::vector<Expr> MergeProduct(const std::vector<Expr>& _p, const std::vector<Expr>& _q) {
  std::vector<Expr> p, q;
  for (auto& e : _p) p.push_back(AutoSimplify(e));
  for (auto& e : _q) q.push_back(AutoSimplify(e));

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

std::vector<Expr> SimplifyProductRec(const std::vector<Expr>& _operands) {
  CHECK_GE(_operands.size(), 2);
  std::vector<Expr> operands;
  for (auto& e : _operands) operands.push_back(AutoSimplify(e));

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
    auto a = AutoSimplify(operands[0]);
    auto b = AutoSimplify(operands[1]);

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
    auto p0 = AutoSimplify(operands[0]);
    auto w  = SimplifyProductRec(Rest(operands));
    if (p0.As<Product>()) {
      return MergeProduct(p0->operands, w);
    } else {
      return MergeProduct({p0}, w);
    }
  }

  return operands;
}

Expr SimplifyProduct(Expr a) {
  a = SumOrProductGetSingleElementsRec(a);
  // We reuse the Mul node for production.
  auto* prod = a.As<Product>();
  if (!prod) return a;

  const auto& _operands = prod->operands();
  std::vector<Expr> operands;
  for (auto& e : _operands) operands.push_back(AutoSimplify(e));
#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& v : operands) {
      ss << v << " ";
    }
    VLOG(4) << "operands: " << ss.str();
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

Expr SimplifySum(Expr u) {
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
std::vector<Expr> MergeSum(const std::vector<Expr>& _p, const std::vector<Expr>& _q) {
  std::vector<Expr> p, q;
  for (auto& e : _p) {
    p.push_back(AutoSimplify(e));
  }
  for (auto& e : _q) {
    q.push_back(AutoSimplify(e));
  }

#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& x : p) ss << x << " ";

    VLOG(3) << "MergeSum p(" << ss.str() << ")";
    ss.str("");

    for (auto& x : q) ss << x << " ";
    VLOG(3) << "MergeSum q(" << ss.str() << ")";
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
std::vector<Expr> SimplifySumRec(const std::vector<Expr>& _operands) {
  CHECK_GE(_operands.size(), 2UL);

  std::vector<Expr> operands;
  for (auto& e : _operands) operands.push_back(AutoSimplify(e));

#ifdef CINN_DEBUG
  {
    std::stringstream ss;
    for (auto& o : operands) {
      ss << o.node_type() << " " << o << " ";
    }
    VLOG(3) << "SimplifySumRec operands: " << ss.str();
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
          return {AutoSimplify(Mod::Make(Sum::Make({am->a(), bm->a()}), am->b()))};
        }
      }
    }

    // case 3
    // Here is different from SimplifySumRec, to deal with cases like 3x + (-2x) = 2x
    if (ProductGetNonConstantPart(a) == ProductGetNonConstantPart(b)) {
      VLOG(3) << "a " << a;
      VLOG(3) << "b " << b;
      Expr s = SimplifySum(Sum::Make({ProductGetConstantPart(a), ProductGetConstantPart(b)}));
      Expr p = Product::Make({s, ProductGetNonConstantPart(a)});
      return {AutoSimplify(p)};
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

Expr SimplifyMod(Expr u) {
  auto* node = u.As<Mod>();
  CHECK(node);

  auto a = AutoSimplify(node->a());
  auto b = AutoSimplify(node->b());

  auto* ai = a.As<IntImm>();
  auto* bi = b.As<IntImm>();
  // 7 % 3
  if (ai && bi) {
    return make_const(ai->type(), ai->value % bi->value);
  }

  // x % 1 = 0
  if (bi && bi->value == 1) return make_const(bi->type(), 0);

  // 2x % 2 = 0
  if (bi) {
    auto* ap = a.As<Product>();
    if (ap && ap->operand(0).As<IntImm>()) {
      if (ap->operand(0).As<IntImm>()->value % bi->value == 0) return make_const(ap->type(), 0);
    }
  }

  if (ai && (ai->value == 0 || ai->value == 1)) return a;

  // (x+y) % 2 = x%2 + y%2
  if (a.As<Sum>()) {
    std::vector<Expr> sum_args;
    for (auto& v : a->operands) {
      sum_args.push_back(Mod::Make(v, b));
    }
    return AutoSimplify(Sum::Make(sum_args));
  }

  return Mod::Make(a, b);
}

}  // namespace detail

Expr AutoSimplify(Expr u) {
  u = detail::SumOrProductGetSingleElementsRec(u);

  if (u.is_constant() || u.As<_Var_>()) return u;

  if (u.As<FracOp>()) {
    return detail::SimplifyRationalNumber(u);
  }

  if (u.As<Product>()) {
    return detail::SumOrProductGetSingleElementsRec(detail::SimplifyProduct(u));
  }

  if (u.As<Sum>()) {
    return detail::SumOrProductGetSingleElementsRec(detail::SimplifySum(u));
  }

  if (u.As<Mod>()) {
    return detail::SumOrProductGetSingleElementsRec(detail::SimplifyMod(u));
  }

  return u;
}

}  // namespace common
}  // namespace cinn

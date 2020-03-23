#include "cinn/common/arithmatic.h"

#include <map>
#include <numeric>
#include <set>
#include <string>

#include "cinn/common/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace common {

using utils::GetStreamCnt;
using utils::Join;
using utils::Replace;
using utils::Split;
using namespace ir;  // NOLINT

#ifdef As
#undef As
#endif

std::string ExprToGinacConerter::Repr(const ir::Expr& expr) {
  auto* load_n      = expr.As<Load>();
  auto* var_n       = expr.As<_Var_>();
  auto* broadcast_n = expr.As<Broadcast>();
  auto* mod_n       = expr.As<Mod>();
  if (load_n || broadcast_n || mod_n) {
    std::string repr = GetStreamCnt(expr);
    Replace(&repr, "[", "lsq_");
    Replace(&repr, "]", "_rsq");
    Replace(&repr, "(", "lb_");
    Replace(&repr, ")", "_rb");
    Replace(&repr, "+", "_add_");
    Replace(&repr, "-", "_sub_");
    Replace(&repr, ":", "_ref_");
    Replace(&repr, "*", "_mul_");
    Replace(&repr, "/", "_div_");
    // remove the spaces
    auto fields = utils::Split(repr, " ");
    repr        = utils::Join(fields, "_");
    return repr;
  } else if (var_n) {
    return utils::GetStreamCnt(expr);
  }
  return "";
}

void ExprToGinacConerter::RecordExpr(const ir::Expr& expr) { repr_to_expr_[Repr(expr)] = expr; }

GiNaC::ex ExprToGinacConerter::BuildHelper(ir::Expr expr) {
  auto* load_n      = expr.As<Load>();
  auto* var_n       = expr.As<_Var_>();
  auto* int_n       = expr.As<IntImm>();
  auto* float_n     = expr.As<FloatImm>();
  auto* add_n       = expr.As<Add>();
  auto* sub_n       = expr.As<Sub>();
  auto* mul_n       = expr.As<Mul>();
  auto* div_n       = expr.As<Div>();
  auto* minus_n     = expr.As<Minus>();
  auto* broadcast_n = expr.As<Broadcast>();
  auto* mod_n       = expr.As<Mod>();

  if (load_n || var_n || broadcast_n || mod_n) {
    RecordExpr(expr);
    std::string repr = Repr(expr);
    return CreateGinacSymbol(repr);
  } else if (int_n) {
    return int_n->value;
  } else if (float_n) {
    return float_n->value;
  } else if (add_n) {
    auto a = BuildHelper(add_n->a());
    auto b = BuildHelper(add_n->b());
    return (a + b) * 1;
  } else if (sub_n) {
    return (BuildHelper(sub_n->a()) - BuildHelper(sub_n->b()));
  } else if (mul_n) {
    return (BuildHelper(mul_n->a()) * BuildHelper(mul_n->b()));
  } else if (div_n) {
    return (BuildHelper(div_n->a()) / BuildHelper(div_n->b()));
  } else if (minus_n) {
    return -BuildHelper(minus_n->v());
  } else {
    NOT_IMPLEMENTED
  }
}

GiNaC::ex ExprToGinacConerter::operator()(Expr expr) {
  // TODO(Superjomn) Replace this with common::IsPureMath(
  auto complex_nodes = CollectIRNodes(expr, [](const Expr* n) {
    return n->As<Block>() ||    //
           n->As<PolyFor>() ||  //
           n->As<Min>() ||      //
           n->As<Max>() ||      //
           n->As<EQ>() ||       //
           n->As<NE>() ||       //
           n->As<LT>() ||       //
           n->As<LE>() ||       //
           n->As<GT>() ||       //
           n->As<GE>() ||       //
           n->As<And>() ||      //
           n->As<Or>() ||       //
           n->As<Not>() ||      //
           n->As<Let>() ||      //
           n->As<Call>() ||     //
           n->As<Select>() ||   //
           n->As<Store>() ||    //
           n->As<Alloc>() ||    //
           n->As<Free>() ||     //
           n->As<IfThenElse>();
  });

  for (auto& node : complex_nodes) {
    LOG(INFO) << "complex nodes: " << node;
  }
  CHECK(complex_nodes.empty())
      << "Ginac converter can only deal with simple math expression, but get some complex nodes" << expr;

  return BuildHelper(expr);
}

GiNaC::symbol ExprToGinacConerter::CreateGinacSymbol(const std::string& repr) {
  CHECK(!repr.empty());
  auto it = repr_to_ginac_.find(repr);
  if (it != repr_to_ginac_.end()) return it->second;

  GiNaC::symbol x(repr);
  repr_to_ginac_[repr] = x;
  return x;
}

GiNaC::symbol ExprToGinacConerter::CreateGinacSymbol(const ir::Expr& var) {
  CHECK(var.As<_Var_>());
  return CreateGinacSymbol(Repr(var));
}

class GiNaCToExprVisitor : public GiNaC::symbol::visitor,
                           public GiNaC::numeric::visitor,
                           public GiNaC::add::visitor,
                           public GiNaC::mul::visitor,
                           public GiNaC::power::visitor,
                           public GiNaC::basic::visitor,
                           public GiNaC::visitor {
  std::map<std::string, ir::Expr>& repr_to_expr;
  ir::Expr cur;

 public:
  explicit GiNaCToExprVisitor(std::map<std::string, ir::Expr>& repr_to_expr) : repr_to_expr(repr_to_expr) {}

  Expr operator()(GiNaC::ex ex) {
    ex.accept(*this);
    return cur;
  }

  void visit(const GiNaC::symbol& node) override {
    auto it = repr_to_expr.find(node.get_name());
    CHECK(it != repr_to_expr.end()) << "node [" << node.get_name() << "] not found";
    cur = it->second;
  }

  void visit(const GiNaC::numeric& node) override {
    if (node.is_integer()) {
      cur = Expr(static_cast<int>(node.to_int()));
    } else {
      cur = Expr(static_cast<float>(node.to_double()));
    }
  }
  void visit(const GiNaC::add& node) override {
    node.op(0).accept(*this);
    Expr res = cur;

    for (int i = 1; i < node.nops(); i++) {
      node.op(i).accept(*this);
      res = res + cur;
    }

    cur = res;
  }

  void visit(const GiNaC::power& node) override {
    node.op(0).accept(*this);
    Expr a = cur;
    node.op(1).accept(*this);

    auto* intv = cur.As<IntImm>();
    CHECK(intv);
    CHECK_EQ(intv->value, -1);

    cur = Div::Make(Expr(1), a);
  }

  void visit(const GiNaC::mul& node) override {
    node.op(0).accept(*this);
    Expr res = cur;

    for (int i = 1; i < node.nops(); i++) {
      node.op(i).accept(*this);
      res = res * cur;
    }

    cur = res;
  }
  void visit(const GiNaC::basic& basic) override { NOT_IMPLEMENTED }
};

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

Expr ExprToGinacConerter::GinacToExpr(const GiNaC::ex& ex) {
  GiNaCToExprVisitor visitor(repr_to_expr_);
  return visitor(ex);
}

bool IsPureMath(Expr expr) {
  std::set<IrNodeTy> valid_node_tys({
      IrNodeTy ::_Var_,
      IrNodeTy ::IntImm,
      IrNodeTy ::FloatImm,
      IrNodeTy ::Add,
      IrNodeTy ::Sub,
      IrNodeTy ::Div,
      IrNodeTy ::Mul,
      IrNodeTy ::Minus,
  });

  auto complex_nodes = ir::CollectIRNodes(expr, [&](const Expr* n) { return !valid_node_tys.count(n->node_type()); });
  return complex_nodes.empty();
}

bool MathContainsSymbol(Expr expr, Var symbol) {
  // Use diff(expr, x) and check the result is not zero.
  ExprToGinacConerter expr_converter;
  auto expr_ex = expr_converter(expr);
  if (!expr_converter.HasSymbol(symbol->name)) return false;
  return !ginac::diff(expr_ex, expr_converter.GetSymbol(symbol->name)).is_zero();
}

// lhs >= rhs.
std::tuple<Expr, bool /*positive*/> Solve(Expr lhs, Expr rhs, Var var) {
  ExprToGinacConerter converter;
  auto lhs_ex = converter(lhs);
  auto rhs_ex = converter(rhs);
  ginac::lst eqs{lhs_ex == rhs_ex};
  const auto& symbol = converter.GetSymbol(var->name);
  ginac::lst vars{symbol};
  ginac::ex res = ginac::lsolve(eqs, vars);

  CHECK_EQ(res.nops(), 1);
  auto item = res.op(0);
  CHECK_EQ(item.nops(), 2);
  Expr value = converter.GinacToExpr(item.op(1));

  // tell the symbol
  auto diff     = lhs_ex - rhs_ex;
  auto diff_res = ginac::diff(diff, symbol);
  CHECK(!diff_res.is_zero());

  /*
  struct Visitor : public ginac::visitor, public GiNaC::numeric::visitor {
    int v = std::numeric_limits<int>::min();

    void operator()(GiNaC::ex ex) { ex.accept(*this); }
    void visit(const GiNaC::numeric& node) override {
      if (node.is_positive()) v = 1;
      else v = -1;
    }
  };
  Visitor visitor;
  visitor(diff_res);

  CHECK_NE(visitor.v, std::numeric_limits<int>::min()) << "the diff result should be a integer";
  CHECK_NE(visitor.v, 0) << "the diff result should not be zero";
   */

  return std::make_tuple(value, diff_res > 0);
}

bool MathIsZero(Expr expr) {
  if (!IsPureMath(expr)) return false;
  ExprToGinacConerter converter;

  auto ex = converter(expr);
  return ex.is_zero();
}

//////// All the following symbolic computation methods are implemented referencing to the book <Computer Algegra and
/// Symbolic Computation - Joel S. Cohen>

void SimplifyRNE(Expr* u) {
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
    int dv = ni->value;

    if (Irem(nv, dv) == 0)
      return Expr(make_const(u.type(), Iquot(nv, dv)));
    else {
      int g = gcd(nv, dv);
      if (dv > 0) {
        return FracOp::Make(make_const(Iquot(nv, g)), make_const(Iquot(dv, g)));
      } else {
        return FracOp::Make(make_const(Iquot(-nv, g)), make_const(Iquot(-dv, g)));
      }
    }
  }
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
    Expr p = SimplifyProduct(Mul::Make(n, s));
    if (p.As<IntImm>()) {
      return SimplifyIntegerPower(Power::Make(r, p));
    } else {
      return Power::Make(r, p);
    }
  }

  return u;
}

Expr SimplifyPower(Expr u) {
  auto* node    = u.As<Power>();
  auto* int_v   = node->a().As<IntImm>();
  auto* float_v = node->a().As<FloatImm>();

  // 0^x = 0
  if ((int_v && int_v->value == 0) || (float_v && float_v->value == 0)) {
    return make_const(node->a().type(), 0);
  }
  // 1^x = 1
  if ((int_v && int_v->value == 1) || (float_v && float_v->value == 1.f)) {
    return make_const(node->a().type(), 1);
  }

  if (node->b().As<IntImm>()) {
    return SimplifyIntegerPower(u);
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

  // O-7, if a is an integer or fraction and v is any other type, 1 < x
  if (a.As<IntImm>() || a.As<FloatImm>() || a.As<FracOp>()) {
    if (!(b.As<IntImm>() || b.As<FloatImm>() || b.As<FracOp>())) return true;
  }

  // O-8, if a is a product, v is a power, sum, fractional, or symbol
  {
    auto* ap = a.As<Product>();

    if (ap && (b.As<Power>() || b.As<Sum>() || b.As<Call>() || b.As<_Var_>())) {
      return operator()(a, Product::Make({b}));
    }
  }

  // O-9, if a is a power, b is a sum, function or symbol
  {
    if (a.As<Power>()) {
      if (b.As<Add>() || b.As<Call>() || b.As<_Var_>()) {
        return operator()(a, Power::Make(b, make_const(1)));
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

std::vector<Expr> SimplifyProductRec(const std::vector<Expr>& oprands) {
  CHECK_GE(oprands.size(), 2);

  if (oprands.size() == 2) {
    auto* ai = oprands[0].As<IntImm>();
    auto* af = oprands[0].As<FloatImm>();
    auto* bi = oprands[1].As<IntImm>();
    auto* bf = oprands[1].As<FloatImm>();

    // both are constants
    if (ai || af) {
      float av = ai ? ai->value : af->value;

      if (bi || bf) {
        float bv = bi ? bi->value : bf->value;

        float mul = av * bv;
        if (mul == 1) return {};
        return {make_const(oprands[0].type(), av * bv)};
      }
    }

    // x*1 -> a
    if (ai && ai->value == 1) return {oprands[1]};
    if (af && af->value == 1.f) return {oprands[1]};
    // 1*x -> x
    if (bi && bi->value == 1) return {oprands[0]};
    if (bf && bf->value == 1.f) return {oprands[0]};

    // Skip the exponent way
  }
}

Expr SimplifyProduct(Expr a) {
  return a;
  // We reuse the Mul node for production.
  auto* prod = a.As<Product>();
  CHECK(prod);
  const auto& operands = prod->operands();

  // 0*x... = 0
  for (auto& opr : operands) {
    auto* opri = opr.As<IntImm>();
    auto* oprf = opr.As<FloatImm>();
    if (opri && opri->value == 0) return make_const(a.type(), 0);
    if (oprf && oprf->value == 0) return make_const(a.type(), 0);
  }

  // prod(x) = x, single number.
  if (operands.size() == 1) {
    return operands[0];
  }
}

Expr EvaluateSum(Expr v, Expr w) { return Expr(); }
Expr EvaluateProd(Expr v, Expr w) { return Expr(); }

}  // namespace detail

}  // namespace common
}  // namespace cinn

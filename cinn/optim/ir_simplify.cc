#include "cinn/optim/ir_simplify.h"
#include <ginac/ginac.h>
#include <glog/logging.h>
#include <map>
#include <string>
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using utils::GetStreamCnt;
using utils::Replace;

/**
 * Helper to convert cinn::Expr to GiNaC::expr for some symbolic math analysis.
 */
struct ExprToGinacConerter {
  GiNaC::ex operator()(Expr expr);

  std::string Repr(const Expr& expr);

  //! Convert GiNaC ex to CINN expression.
  Expr GinacToExpr(const GiNaC::ex& ex);

  GiNaC::symbol CreateGinacSymbol(const std::string& repr);
  GiNaC::symbol CreateGinacSymbol(const ir::Expr& var);

 private:
  GiNaC::ex BuildHelper(ir::Expr expr);

  void RecordExpr(const ir::Expr& expr);

 private:
  std::map<std::string, ir::Expr> repr_to_expr_;
  std::map<std::string, GiNaC::symbol> repr_to_ginac_;
};

std::string ExprToGinacConerter::Repr(const Expr& expr) {
  auto* load_n = expr.As<Load>();
  auto* var_n  = expr.As<_Var_>();
  CHECK(load_n || var_n);
  if (load_n) {
    std::string repr = GetStreamCnt(expr);
    Replace(&repr, "[", "lsq_");
    Replace(&repr, "]", "_rsq");
    Replace(&repr, "(", "lb_");
    Replace(&repr, ")", "_rb");
    Replace(&repr, "+", "_add_");
    Replace(&repr, "-", "_sub_");
    Replace(&repr, "*", "_mul_");
    Replace(&repr, "/", "_div_");
    // remove the spaces
    auto fields = utils::Split(repr, " ");
    repr        = utils::Join(fields, "_");
    return repr;
  } else if (var_n) {
    return utils::GetStreamCnt(expr);
  }
}

void ExprToGinacConerter::RecordExpr(const ir::Expr& expr) {
  CHECK(expr.As<Load>() || expr.As<_Var_>());
  repr_to_expr_[Repr(expr)] = expr;
}

GiNaC::ex ExprToGinacConerter::BuildHelper(ir::Expr expr) {
  auto* load_n  = expr.As<Load>();
  auto* var_n   = expr.As<_Var_>();
  auto* int_n   = expr.As<IntImm>();
  auto* float_n = expr.As<FloatImm>();
  auto* add_n   = expr.As<Add>();
  auto* sub_n   = expr.As<Sub>();
  auto* mul_n   = expr.As<Mul>();
  auto* div_n   = expr.As<Div>();
  auto* minus_n = expr.As<Minus>();

  if (load_n || var_n) {
    RecordExpr(expr);
    std::string repr = Repr(expr);
    return CreateGinacSymbol(repr);
  } else if (int_n) {
    return int_n->value;
  } else if (float_n) {
    return float_n->value;
  } else if (add_n) {
    auto a = BuildHelper(add_n->a);
    auto b = BuildHelper(add_n->b);
    return (a + b) * 1;
  } else if (sub_n) {
    return (BuildHelper(sub_n->a) - BuildHelper(sub_n->b));
  } else if (mul_n) {
    return (BuildHelper(mul_n->a) * BuildHelper(mul_n->b));
  } else if (div_n) {
    return (BuildHelper(div_n->a) / BuildHelper(div_n->b));
  } else if (minus_n) {
    return -BuildHelper(minus_n->v);
  } else {
    NOT_IMPLEMENTED
  }
}

GiNaC::ex ExprToGinacConerter::operator()(Expr expr) {
  auto complex_nodes = CollectIRNodes(expr, [](const Expr* n) {
    return n->As<Block>() ||       //
           n->As<PolyFor>() ||     //
           n->As<Ramp>() ||        //
           n->As<Min>() ||         //
           n->As<Max>() ||         //
           n->As<EQ>() ||          //
           n->As<NE>() ||          //
           n->As<LT>() ||          //
           n->As<LE>() ||          //
           n->As<GT>() ||          //
           n->As<GE>() ||          //
           n->As<And>() ||         //
           n->As<Or>() ||          //
           n->As<Not>() ||         //
           n->As<Let>() ||         //
           n->As<Call>() ||        //
           n->As<Select>() ||      //
           n->As<Store>() ||       //
           n->As<Alloc>() ||       //
           n->As<Free>() ||        //
           n->As<IfThenElse>() ||  //
           n->As<Broadcast>();
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

Expr ExprToGinacConerter::GinacToExpr(const GiNaC::ex& ex) {
  GiNaCToExprVisitor visitor(repr_to_expr_);
  return visitor(ex);
}

//! Simplify some sub-expression in the `expr`. Due to the simplify strategy just fit several kinds of IR noedes, we
//! partition the original expression to several sub-expression those supported by simplify, and process each of them.
void PartialSimplify(Expr* expr) {
  ExprToGinacConerter converter;
  auto ex = converter(*expr);
  VLOG(4) << "get ex:" << ex;
  *expr = converter.GinacToExpr(ex);
  VLOG(4) << "ex to expr: " << *expr;
}

namespace {

struct SimplifyStoreMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Store* expr, Expr* op) override {
    auto* node = op->As<Store>();
    VLOG(4) << "to simplify Load: " << *op;
    PartialSimplify(&node->index);
    VLOG(4) << "get: " << *op;
  }
};

struct SimplifyLoadMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Load* expr, Expr* op) override {
    auto* node = op->As<Load>();
    VLOG(4) << "to simplify Load: " << *op;
    PartialSimplify(&node->index);
    VLOG(4) << "get: " << *op;
  }
};

//! Simplify the expression but Load.
struct SimplifyButStoreLoadMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Add* expr, Expr* op) override {
    VLOG(4) << "origin: " << *op;
    PartialSimplify(op);
    VLOG(4) << "simplified: " << *op;
  }
  void Visit(const Sub* expr, Expr* op) override {
    VLOG(4) << "origin: " << *op;
    PartialSimplify(op);
    VLOG(4) << "simplified: " << *op;
  }
  void Visit(const Mul* expr, Expr* op) override {
    VLOG(4) << "origin: " << *op;
    PartialSimplify(op);
    VLOG(4) << "simplified: " << *op;
  }
  void Visit(const Div* expr, Expr* op) override {
    VLOG(4) << "origin: " << *op;
    PartialSimplify(op);
    VLOG(4) << "simplified: " << *op;
  }

#undef __
};

}  // namespace

void Simplify(Expr* expr) {
  SimplifyLoadMutator()(expr);
  SimplifyStoreMutator()(expr);
  SimplifyButStoreLoadMutator()(expr);
}

}  // namespace optim
}  // namespace cinn

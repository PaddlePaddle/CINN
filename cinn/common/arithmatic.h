/**
 * This file includes some arithmatic utilities, such as simplifying/solving a math equation/CINN expression.
 */
#pragma once

#include <ginac/ginac.h>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include "cinn/ir/ir.h"

#ifdef As
#undef As
#endif

namespace cinn {
namespace common {

namespace ginac = GiNaC;

//! Tell whether the expression \p expr contains only simple math calculations, like i*32+j is true, while Load(buf,
//! i)+1 is not due to the Load Node is not math related.
bool IsPureMath(Expr expr);

//! Tell whether the expression \p expr contains the expression \symbol, e.g. i*32+32 contains `i`, it also contains
//! `i+1`.
bool MathContainsSymbol(Expr expr, Var symbol);

//! Solve the equation \p lhs == \p rhs on symbol \p symbol.
std::tuple<Expr, bool /*positive*/> Solve(Expr lhs, Expr rhs, Var symbol);

/**
 * Helper to convert cinn::Expr to GiNaC::expr for some symbolic math analysis.
 */
struct ExprToGinacConerter {
  //! Convert CINN expression \p expr to GiNaC ex.
  ginac::ex operator()(Expr expr);

  //! Convert GiNaC ex back to CINN expression, should call operator() first.
  Expr GinacToExpr(const GiNaC::ex& ex);

  const ginac::symbol& GetSymbol(const std::string& name) const { return repr_to_ginac_.at(name); }

 private:
  std::string Repr(const Expr& expr);
  ginac::symbol CreateGinacSymbol(const std::string& repr);
  ginac::symbol CreateGinacSymbol(const ir::Expr& var);

  ginac::ex BuildHelper(ir::Expr expr);

  void RecordExpr(const ir::Expr& expr);

 private:
  std::map<std::string, ir::Expr> repr_to_expr_;
  std::map<std::string, ginac::symbol> repr_to_ginac_;
};

}  // namespace common
}  // namespace cinn

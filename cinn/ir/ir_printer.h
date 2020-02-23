#pragma once
#include <string>
#include <vector>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

struct IrPrinter : public IRVisitor {
  explicit IrPrinter(std::ostream &os) : os_(os) {}

  //! Emit an expression on the output stream.
  void Print(Expr e);
  //! Emit a expression list with , splitted.
  void Print(const std::vector<Expr> &exprs, const std::string &splitter = ", ");
  //! Emit a binary operator
  template <typename IRN>
  void PrintBinaryOp(const std::string &op, const BinaryOpNode<IRN> *x) {
    os_ << "(";
    Print(x->a);
    os_ << " " + op + " ";
    Print(x->b);
    os_ << ")";
  }

  //! Prefix the current line with `indent_` spaces.
  void DoIndent();
  //! Increase the indent size.
  void IncIndent();
  //! Decrease the indent size.
  void DecIndent();

  std::ostream &os() { return os_; }

  void Visit(const IntImm *x) override;
  void Visit(const UIntImm *x) override;
  void Visit(const FloatImm *x) override;
  void Visit(const Add *x) override;
  void Visit(const Sub *x) override;
  void Visit(const Mul *x) override;
  void Visit(const Div *x) override;
  void Visit(const Mod *x) override;
  void Visit(const EQ *x) override;
  void Visit(const NE *x) override;
  void Visit(const LT *x) override;
  void Visit(const LE *x) override;
  void Visit(const GT *x) override;
  void Visit(const GE *x) override;
  void Visit(const And *x) override;
  void Visit(const Or *x) override;
  void Visit(const Not *x) override;
  void Visit(const Min *x) override;
  void Visit(const Max *x) override;
  void Visit(const Minus *x) override;
  void Visit(const For *x) override;
  void Visit(const PolyFor *x) override;
  void Visit(const IfThenElse *x) override;
  void Visit(const Block *x) override;
  void Visit(const Call *x) override;
  void Visit(const Cast *x) override;
  void Visit(const Module *x) override;
  void Visit(const _Var_ *x) override;
  void Visit(const Alloc *x) override;
  void Visit(const Select *x) override;
  void Visit(const Load *x) override;
  void Visit(const Store *x) override;
  void Visit(const Free *x) override;
  void Visit(const _Range_ *x) override;
  void Visit(const _IterVar_ *x) override {}
  void Visit(const _Buffer_ *x) override;
  void Visit(const _Tensor_ *x) override;

 private:
  std::ostream &os_;
  uint16_t indent_{};
  const int indent_unit{2};
};

std::ostream &operator<<(std::ostream &os, Expr a);

}  // namespace ir
}  // namespace cinn

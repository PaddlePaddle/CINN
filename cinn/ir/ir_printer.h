#pragma once
#include <string>
#include <vector>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {

namespace lang {
class Module;
class LoweredFunc;
}  // namespace lang

namespace ir {

struct IrPrinter : public IRVisitor {
  explicit IrPrinter(std::ostream &os) : os_(os) {}

  //! Emit an expression on the output stream.
  void Print(Expr e);
  //! Emit a expression list with , splitted.
  void Print(const std::vector<Expr> &exprs, const std::string &splitter = ", ");
  //! Emit a binary operator
  template <typename IRN>
  void PrintBinaryOp(const std::string &op, const BinaryOpNode<IRN> *x);

  //! Prefix the current line with `indent_` spaces.
  void DoIndent();
  //! Increase the indent size.
  void IncIndent();
  //! Decrease the indent size.
  void DecIndent();

  std::ostream &os() { return os_; }

#define __(op__) void Visit(const op__ *x) override;
  NODETY_FORALL(__)
#undef __

 private:
  std::ostream &os_;
  uint16_t indent_{};
  const int indent_unit{2};
};

std::ostream &operator<<(std::ostream &os, Expr a);
std::ostream &operator<<(std::ostream &os, const std::vector<Expr> &a);
std::ostream &operator<<(std::ostream &os, const lang::Module &m);

template <typename IRN>
void IrPrinter::PrintBinaryOp(const std::string &op, const BinaryOpNode<IRN> *x) {
  os_ << "(";
  Print(x->a());
  os_ << " " + op + " ";
  Print(x->b());
  os_ << ")";
}

}  // namespace ir
}  // namespace cinn

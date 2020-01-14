#pragma once
#include <string>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

struct IrPrinter : public IrVisitor {
  explicit IrPrinter(std::ostream &os) : os_(os) {}

  //! Emit an expression on the output stream.
  void Print(Expr e);
  //! Emit a statement on the output stream.
  void Print(Stmt s);
  //! Emit a binary operator
  template <typename IRN>
  void PrintBinaryOp(const std::string &op, BinaryOpNode<IRN> *x) {
    os_ << "(";
    Print(x->a);
    os_ << " " + op + " ";
    Print(x->b);
    os_ << ")";
  }

  //! Prefix the current line with `indent_` spaces.
  void DoIndent();
  //! Increase the indent size.
  void IncIndent() { ++indent_; }
  //! Decrease the indent size.
  void DescIndent() { --indent_; }

  void Visit(IntImm *x) override;
  void Visit(UIntImm *x) override;
  void Visit(FloatImm *x) override;
  void Visit(Add *x) override;
  void Visit(Sub *x) override;
  void Visit(Mul *x) override;
  void Visit(Div *x) override;
  void Visit(Mod *x) override;
  void Visit(EQ *x) override;
  void Visit(NE *x) override;
  void Visit(LT *x) override;
  void Visit(LE *x) override;
  void Visit(GT *x) override;
  void Visit(GE *x) override;
  void Visit(And *x) override;
  void Visit(Or *x) override;
  void Visit(Not *x) override;
  void Visit(Min *x) override;
  void Visit(Max *x) override;
  void Visit(For *x) override;
  void Visit(IfThenElse *x) override;
  void Visit(Block *x) override;
  void Visit(Call *x) override;
  void Visit(Cast *x) override;
  void Visit(Module *x) override;
  void Visit(Variable *x) override;
  void Visit(Alloc *x) override;

 private:
  std::ostream &os_;
  uint16_t indent_{};
};

}  // namespace ir
}  // namespace cinn

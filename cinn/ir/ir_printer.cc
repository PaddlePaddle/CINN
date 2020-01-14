#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace ir {

void IrPrinter::Print(Expr e) { e.Accept(reinterpret_cast<IRVisitor *>(this)); }
void IrPrinter::Print(Stmt s) { s.Accept(reinterpret_cast<IRVisitor *>(this)); }

void IrPrinter::Visit(IntImm *x) { os_ << x->value; }
void IrPrinter::Visit(UIntImm *x) { os_ << x->value; }
void IrPrinter::Visit(FloatImm *x) { os_ << x->value; }
void IrPrinter::Visit(Add *x) { PrintBinaryOp("+", x); }
void IrPrinter::Visit(Sub *x) { PrintBinaryOp("-", x); }
void IrPrinter::Visit(Mul *x) { PrintBinaryOp("*", x); }
void IrPrinter::Visit(Div *x) { PrintBinaryOp("/", x); }
void IrPrinter::Visit(Mod *x) { PrintBinaryOp("%", x); }
void IrPrinter::Visit(EQ *x) { PrintBinaryOp("==", x); }
void IrPrinter::Visit(NE *x) { PrintBinaryOp("!=", x); }
void IrPrinter::Visit(LT *x) { PrintBinaryOp("<", x); }
void IrPrinter::Visit(LE *x) { PrintBinaryOp("<=", x); }
void IrPrinter::Visit(GT *x) { PrintBinaryOp(">", x); }
void IrPrinter::Visit(GE *x) { PrintBinaryOp("<=", x); }
void IrPrinter::Visit(And *x) { PrintBinaryOp("and", x); }
void IrPrinter::Visit(Or *x) { PrintBinaryOp("or", x); }
void IrPrinter::Visit(Not *x) {
  os_ << "!";
  Print(x->v);
}
void IrPrinter::Visit(Min *x) {
  os_ << "min(";
  Print(x->a);
  os_ << ", ";
  Print(x->b);
  os_ << ")";
}
void IrPrinter::Visit(Max *x) {
  os_ << "max(";
  Print(x->a);
  os_ << ", ";
  Print(x->b);
  os_ << ")";
}
void IrPrinter::Visit(For *x) {
  DoIndent();
  os_ << "for(";
  Print(x->min);
  os_ << ", ";
  Print(x->extent);
  os_ << ")";
}
void IrPrinter::Visit(IfThenElse *x) {
  DoIndent();
  os_ << "if (";
  Print(x->condition);
  os_ << ")";
  Print(x->true_case);
  os_ << "\n";

  if (x->false_case.defined()) {
    os_ << "else ";
    Print(x->false_case);
    os_ << "\n";
  }
}
void IrPrinter::Visit(Block *x) {
  DoIndent();
  os_ << "{\n";

  IncIndent();
  for (auto &s : x->stmts) {
    DoIndent();
    Print(s);
    os_ << "\n";
  }
  DescIndent();

  DoIndent();
  os_ << "}";
}
void IrPrinter::Visit(Call *x) {}
void IrPrinter::Visit(Cast *x) {}
void IrPrinter::Visit(Module *x) {}
void IrPrinter::Visit(Variable *x) { os_ << x->name; }
void IrPrinter::Visit(Alloc *x) {}

void IrPrinter::DoIndent() {
  for (int i = 0; i < indent_; i++) os_ << ' ';
}

}  // namespace ir
}  // namespace cinn

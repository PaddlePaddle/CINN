#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace ir {

void IrPrinter::Print(Expr e) { e.Accept(reinterpret_cast<IrVisitor *>(this)); }
void IrPrinter::Print(Stmt s) { s.Accept(reinterpret_cast<IrVisitor *>(this)); }
void IrPrinter::Print(const std::vector<Expr> &exprs, const std::string &splitter) {
  for (int i = 0; i < exprs.size() - 1; i++) {
    Print(exprs[i]);
    os_ << ", ";
  }
  if (exprs.size() > 1) Print(exprs.back());
}

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
void IrPrinter::Visit(Alloc *x) {
  os_ << "alloc(" << x->buffer_var->name << ", ";
  Print(x->extents);
  os_ << ")";
}
void IrPrinter::Visit(Select *x) {
  os_ << "select(";
  Print(x->condition);
  os_ << ", ";
  Print(x->true_value);
  os_ << ", ";
  Print(x->false_value);
  os_ << ")";
}
void IrPrinter::Visit(Load *x) {
  os_ << x->buffer_var->name << "[";
  Print(x->index);
  os_ << "]";
}
void IrPrinter::Visit(Store *x) {
  os_ << x->buffer_var->name << "[";
  Print(x->index);
  os_ << "] = ";
  Print(x->value);
}
void IrPrinter::Visit(Free *x) { os_ << "free(" << x->var->name << ")"; }

void IrPrinter::DoIndent() {
  for (int i = 0; i < indent_; i++) os_ << ' ';
}

}  // namespace ir
}  // namespace cinn

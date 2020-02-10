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

void IrPrinter::Visit(const IntImm *x) { os_ << x->value; }
void IrPrinter::Visit(const UIntImm *x) { os_ << x->value; }
void IrPrinter::Visit(const FloatImm *x) { os_ << x->value; }
void IrPrinter::Visit(const Add *x) { PrintBinaryOp("+", x); }
void IrPrinter::Visit(const Sub *x) { PrintBinaryOp("-", x); }
void IrPrinter::Visit(const Mul *x) { PrintBinaryOp("*", x); }
void IrPrinter::Visit(const Div *x) { PrintBinaryOp("/", x); }
void IrPrinter::Visit(const Mod *x) { PrintBinaryOp("%", x); }
void IrPrinter::Visit(const EQ *x) { PrintBinaryOp("==", x); }
void IrPrinter::Visit(const NE *x) { PrintBinaryOp("!=", x); }
void IrPrinter::Visit(const LT *x) { PrintBinaryOp("<", x); }
void IrPrinter::Visit(const LE *x) { PrintBinaryOp("<=", x); }
void IrPrinter::Visit(const GT *x) { PrintBinaryOp(">", x); }
void IrPrinter::Visit(const GE *x) { PrintBinaryOp("<=", x); }
void IrPrinter::Visit(const And *x) { PrintBinaryOp("and", x); }
void IrPrinter::Visit(const Or *x) { PrintBinaryOp("or", x); }
void IrPrinter::Visit(const Not *x) {
  os_ << "!";
  Print(x->v);
}
void IrPrinter::Visit(const Min *x) {
  os_ << "min(";
  Print(x->a);
  os_ << ", ";
  Print(x->b);
  os_ << ")";
}
void IrPrinter::Visit(const Max *x) {
  os_ << "max(";
  Print(x->a);
  os_ << ", ";
  Print(x->b);
  os_ << ")";
}
void IrPrinter::Visit(const For *x) {
  DoIndent();
  os_ << "for(";
  Print(x->min);
  os_ << ", ";
  Print(x->extent);
  os_ << ")";
}
void IrPrinter::Visit(const IfThenElse *x) {
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
void IrPrinter::Visit(const Block *x) {
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
void IrPrinter::Visit(const Call *x) {
  os_ << x->name << "(";
  for (int i = 0; i < x->args.size() - 1; i++) {
    Print(x->args[i]);
    os_ << ", ";
  }
  if (x->args.size() > 1) Print(x->args.back());
  os_ << ")";
}
void IrPrinter::Visit(const Cast *x) {}
void IrPrinter::Visit(const Module *x) {}
void IrPrinter::Visit(const _Var_ *x) { os_ << x->name; }
void IrPrinter::Visit(const Alloc *x) {
  os_ << "alloc(" << x->buffer_var->name << ", ";
  Print(x->extents);
  os_ << ")";
}
void IrPrinter::Visit(const Select *x) {
  os_ << "select(";
  Print(x->condition);
  os_ << ", ";
  Print(x->true_value);
  os_ << ", ";
  Print(x->false_value);
  os_ << ")";
}
void IrPrinter::Visit(const Load *x) {
  os_ << x->buffer_var->name << "[";
  Print(x->index);
  os_ << "]";
}
void IrPrinter::Visit(const Store *x) {
  os_ << x->buffer_var->name << "[";
  Print(x->index);
  os_ << "] = ";
  Print(x->value);
}
void IrPrinter::Visit(const Free *x) { os_ << "free(" << x->var->name << ")"; }

void IrPrinter::DoIndent() {
  for (int i = 0; i < indent_; i++) os_ << ' ';
}

void IrPrinter::Visit(const _Range_ *x) {
  os_ << "Range(min=";
  Print(x->min);
  os_ << ", "
      << "extent=";
  Print(x->extent);
  os_ << ")";
}

void IrPrinter::Visit(const _Buffer_ *x) { os_ << "_Buffer_(" << x->name << ")"; }
void IrPrinter::Visit(const _Tensor_ *x) {
  os_ << "Tensor(";
  for (auto &i : x->shape) {
    Print(i);
    os_ << ",";
  }
  os_ << ")";
}

std::ostream &operator<<(std::ostream &os, Expr a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(a);
  os << ss.str();
  return os;
}

}  // namespace ir
}  // namespace cinn

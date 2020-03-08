#include "cinn/ir/ir_printer.h"

#include <vector>

#include "cinn/ir/lowered_func.h"
#include "cinn/lang/module.h"
#include "cinn/lang/tensor.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

void IrPrinter::Print(Expr e) { IRVisitor::Visit(&e); }
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
void IrPrinter::Visit(const Minus *x) {
  os_ << "-(";
  Print(x->v);
  os_ << ")";
}
void IrPrinter::Visit(const For *x) {
  os_ << "for (";
  Print(x->min);
  os_ << ", ";
  Print(x->extent);
  os_ << ") {\n";

  Print(x->body);

  DoIndent();
  os_ << "}\n";
}

void IrPrinter::Visit(const PolyFor *x) {
  os_ << "poly_for (";
  Print(x->init);
  os_ << ", ";
  Print(x->condition);
  os_ << ", ";
  Print(x->inc);
  os_ << ")\n";

  DoIndent();
  Print(x->body);
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
  os_ << "{\n";

  IncIndent();
  for (int i = 0; i < x->stmts.size() - 1; i++) {
    DoIndent();
    Print(x->stmts[i]);
    os_ << "\n";
  }
  if (x->stmts.size() >= 1) {
    DoIndent();
    Print(x->stmts.back());
  }
  DecIndent();
  os_ << "\n";
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
void IrPrinter::Visit(const Cast *x) {
  os() << x->type();
  os() << "(";
  os() << x->v;
  os() << ")";
}
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
  auto *node = x->tensor.As<ir::_Tensor_>();
  CHECK(node);
  os_ << node->name << "[";
  Print(x->index);
  os_ << "]";
}
void IrPrinter::Visit(const Store *x) {
  auto *tensor_node = x->tensor.As<ir::_Tensor_>();
  CHECK(tensor_node);
  os_ << tensor_node->name << "[";
  Print(x->index);
  os_ << "] = ";
  Print(x->value);
}
void IrPrinter::Visit(const Free *x) { os_ << "free(" << x->var->name << ")"; }

void IrPrinter::DoIndent() { os_ << std::string(indent_, ' '); }
void IrPrinter::IncIndent() { indent_ += indent_unit; }
void IrPrinter::DecIndent() { indent_ -= indent_unit; }

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
  CHECK(!x->shape.empty());
  os_ << "Tensor(";
  for (auto &i : x->shape) {
    Print(i);
    os_ << ",";
  }
  os_ << ")";
}
void IrPrinter::Visit(const _LoweredFunc_ *f) {
  os_ << "function " << f->name << " ";

  std::vector<std::string> arg_names;
  for (auto &arg : f->args) {
    arg_names.push_back(arg.name());
  }
  os_ << "(" << utils::Join(arg_names, ", ") << ")\n";

  Print(f->body);
}
void IrPrinter::Visit(const Let *f) {
  CHECK(f->type().valid());
  os() << f->type() << " ";
  Print(f->value);
  os() << " = ";
  Print(f->body);
}

void IrPrinter::Visit(const _IterVar_ *f) { NOT_IMPLEMENTED }

void IrPrinter::Visit(const Reduce *f){NOT_IMPLEMENTED}

std::ostream &
operator<<(std::ostream &os, Expr a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(a);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const lang::Module &m);

}  // namespace ir
}  // namespace cinn

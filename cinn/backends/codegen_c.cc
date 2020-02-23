#include "cinn/backends/codegen_c.h"

namespace cinn {
namespace backends {

CodeGenC::CodeGenC(std::ostream &os, Target target) : ir::IrPrinter(os), target_(target) {}

void CodeGenC::Compile(const lang::Module &module) {}
void CodeGenC::Compile(const lang::LoweredFunc &function) {
  os() << "void " << function.name;

  // output arguments
  os() << "(";

  auto print_arg = [&](const lang::Argument &arg) {
    if (arg.is_buffer()) {
      os() << "struct cinn_buffer_t *";
    } else if (arg.is_scalar()) {
      os() << PrintType(arg.type) << " ";
      os() << arg.name;
    }
    os() << arg.name;
  };

  for (int i = 0; i < function.args.size() - 1; i++) {
    print_arg(function.args[i]);
    os() << ", ";
  }
  if (function.args.size() >= 1) {
    print_arg(function.args.back());
  }

  os() << ")";

  DoIndent();
  os() << "{\n";

  Print(function.body);

  DoIndent();
  os() << "}";
}
void CodeGenC::Compile(const ir::Buffer &buffer) {}
std::string CodeGenC::PrintType(Type type) { return std::__cxx11::string(); }
void CodeGenC::Visit(const ir::IntImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::UIntImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::FloatImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Add *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Sub *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Mul *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Div *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Mod *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::EQ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::NE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::LT *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::LE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::GT *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::GE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::And *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Or *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Min *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Max *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Minus *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Not *op) {
  os() << "(!";
  IrPrinter::Print(op->v);
  os() << ")";
}
void CodeGenC::Visit(const ir::Cast *op) { PrintCastExpr(op->type(), op->v); }
void CodeGenC::Visit(const ir::For *op) { LOG(FATAL) << "Not Implemented"; }
void CodeGenC::Visit(const ir::PolyFor *op) {
  os() << "for (";
  Print(op->init);
  os() << "; ";
  Print(op->condition);
  os() << "; ";
  Print(op->inc);
  os() << ")";

  Print(op->body);
}
void CodeGenC::Visit(const ir::Select *op) {
  os() << "(";
  os() << "(";
  Print(op->condition);
  os() << ") ? ";
  Print(op->true_value);
  os() << " : ";
  Print(op->false_value);
  os() << ")";
}
void CodeGenC::Visit(const ir::IfThenElse *op) {
  os() << "if (";
  Print(op->condition);
  os() << ")";
  Print(op->true_case);

  if (op->false_case.defined()) {
    os() << "else\n";
    Print(op->false_case);
  }
}
void CodeGenC::Visit(const ir::Block *op) {
  os() << "{\n";

  IncIndent();

  for (int i = 0; i < op->stmts.size() - 1; i++) {
    DoIndent();
    Print(op->stmts[i]);
    os() << ";\n";
  }
  if (op->stmts.size() >= 1) {
    DoIndent();
    Print(op->stmts.back());
    os() << ";";
  }

  DecIndent();
  os() << "\n";
  DoIndent();
  os() << "}";
}
void CodeGenC::Visit(const ir::Call *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Module *op) { NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::_Var_ *op) { os() << op->name; }
void CodeGenC::Visit(const ir::Load *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Store *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Alloc *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Free *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::_Range_ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::_IterVar_ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::_Buffer_ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::_Tensor_ *op) { IrPrinter::Visit(op); }

void CodeGenC::PrintCastExpr(const Type &type, Expr e) {
  os() << PrintType(type) << "(";
  Print(e);
  os() << ")";
}

}  // namespace backends
}  // namespace cinn

#include "cinn/backends/codegen_c.h"

#include "cinn/ir/lowered_func.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/runtime/intrinsic.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace backends {
using namespace utils;

CodeGenC::CodeGenC(std::ostream &os, Target target) : ir::IrPrinter(os), target_(target) {}

void CodeGenC::Compile(const lang::Module &module) {
  PrintFileGuardOpen(module.name());
  PrintIncludes();

  PrintBufferCreation(module->buffers);

  for (auto &func : module.functions()) {
    Compile(func);
  }

  PrintFileGuardClose(module.name());
}
void CodeGenC::Compile(const ir::LoweredFunc &function) {
  Print(function);
  os() << "\n\n";
}
void CodeGenC::Compile(const ir::Buffer &buffer) {
  Print(buffer.CreateExpr());
  os() << "\n";
  os() << "\n";
}

std::string CodeGenC::PrintType(Type type) {
  if (type == Int(8)) {
    return "int8_t";
  }
  if (type == Int(32)) {
    return "int32_t";
  }
  if (type == Int(64)) {
    return "int64_t";
  }
  if (type == Bool()) {
    return "bool";
  }
  if (type == Float(32)) {
    return "float";
  }
  if (type == Float(64)) {
    return "double";
  }

  LOG(ERROR) << type;
  NOT_IMPLEMENTED
}
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
  os() << PrintType(Int(32));
  os() << " " << op->iterator->name;
  os() << " = ";
  Print(op->init);
  os() << "; ";
  Print(op->condition);
  os() << "; ";

  os() << op->iterator->name;
  os() << " += ";
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
void CodeGenC::Visit(const ir::Call *op) {
  if (op->name == runtime::buffer_create) {
    CHECK_EQ(op->args.size(), 1UL);
    os() << "cinn_buffer_t* " << op->args.front();
    os() << " = " << op->name << "()";
  } else if (op->call_type == ir::Call::CallType::Intrinsic) {
    CHECK(!op->args.empty());
    os() << op->name << "(";
    for (int i = 0; i < op->args.size() - 1; i++) {
      os() << op->args[i];
    }
    if (op->args.size() > 0) os() << op->args.back();
    os() << ")";
  } else {
    IrPrinter::Visit(op);
  }
}
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

void CodeGenC::Visit(const ir::_LoweredFunc_ *op) {
  os() << "void " << op->name;

  // output arguments
  os() << "(";

  auto print_arg = [&](const ir::Argument &arg) {
    if (arg.is_buffer()) {
      if (arg.is_input()) {
        os() << "const struct cinn_buffer_t *";
      } else {
        os() << "struct cinn_buffer_t *";
      }
    } else if (arg.is_scalar()) {
      os() << PrintType(arg.type) << " ";
      os() << arg.name;
    } else {
      NOT_IMPLEMENTED
    }
    os() << arg.name;
  };

  for (int i = 0; i < op->args.size() - 1; i++) {
    print_arg(op->args[i]);
    os() << ", ";
  }
  if (op->args.size() >= 1) {
    print_arg(op->args.back());
  }

  os() << ")\n";

  DoIndent();
  // os() << "{\n";

  // allocate output buffer
  Expr allocate_output_buffer_expr = ir::Block::Make(op->alloc_output_buffer_exprs);
  Expr func_body                   = ir::Block::Make({allocate_output_buffer_expr, op->body});

  optim::RemoveNestedBlock(&func_body);

  Print(func_body);

  // DoIndent();
  // os() << "}";
}
void CodeGenC::PrintIncludes() {
  os() << "#include <cinn_runtime.h>\n";
  os() << "#include <stdio.h>\n";
  os() << "\n";
}

void CodeGenC::PrintFileGuardOpen(const std::string &name) {
  os() << utils::StringFormat("#ifndef _%s_CINN_H_\n", Uppercase(name).c_str());
  os() << utils::StringFormat("#define _%s_CINN_H_\n", Uppercase(name).c_str());
  os() << "\n";
}
void CodeGenC::PrintFileGuardClose(const std::string &module_name) {
  os() << utils::StringFormat("#endif  // _%s_CINN_H_\n", Uppercase(module_name).c_str());
}

void CodeGenC::PrintBufferCreation(const std::vector<ir::Buffer> &buffers) {
  for (auto &buffer : buffers) {
    DoIndent();
    Print(buffer.CreateExpr());
    os() << ";\n";
  }
}

void CodeGenC::PrintBufferDestroy(const std::vector<ir::Buffer> &buffers) {
  for (auto &buffer : buffers) {
    DoIndent();
    Print(buffer.DestroyExpr());
    os() << ";\n";
  }
}

}  // namespace backends
}  // namespace cinn

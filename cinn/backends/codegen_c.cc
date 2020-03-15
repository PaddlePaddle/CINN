#include "cinn/backends/codegen_c.h"

#include <fstream>

#include "cinn/ir/lowered_func.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/runtime/intrinsic.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace backends {
using namespace utils;

void CodeGenC::Compile(const lang::Module &module, const Outputs &outputs) {
  if (!outputs.c_header_name.empty()) {
    LOG(WARNING) << "Output C source to file " << outputs.c_header_name;
    auto source = Compile(module, OutputKind::CHeader);
    std::ofstream file(outputs.c_header_name);
    CHECK(file.is_open()) << "failed to open file " << outputs.c_header_name;
    file << source;
    file.close();
  }

  if (!outputs.c_source_name.empty()) {
    LOG(WARNING) << "Output C source to file " << outputs.c_source_name;
    auto source = Compile(module, OutputKind::CImpl);
    std::ofstream file(outputs.c_source_name);
    CHECK(file.is_open()) << "failed to open file " << outputs.c_source_name;
    file << source;
    file.close();
  }
}

CodeGenC::CodeGenC(Target target) : ir::IrPrinter(ss_) {}

std::string CodeGenC::Compile(const lang::Module &module, OutputKind output_kind) {
  ss_.str("");
  if (output_kind == OutputKind::CHeader) {
    GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    PrintIncludes();

    PrintBufferCreation(module->buffers);

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    LOG(FATAL) << "Not supported OutputKind";
  }
  return ss_.str();
}
std::string CodeGenC::Compile(const ir::LoweredFunc &function) {
  Print(function);
  os() << "\n\n";
  return ss_.str();
}

std::string CodeGenC::PrintType(Type type) {
  std::string str;
  if (type.is_cpp_const()) {
    str = "const ";
  }

  if (type.is_int(8)) {
    str += "int8_t";
  } else if (type.is_int(32)) {
    str += "int32_t";
  } else if (type.is_int(64)) {
    str += "int64_t";
  } else if (type.is_bool()) {
    str += "bool";
  } else if (type.is_float(32)) {
    str += "float";
  } else if (type.is_float(64)) {
    str += "double";
  } else {
    LOG(ERROR) << type;
    NOT_IMPLEMENTED
  }

  if (type.is_cpp_handle()) {
    str += "*";
  }
  return str;
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
  os() << ") ";

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
  os() << ") {\n";

  if (!op->true_case.As<ir::Block>()) IncIndent();
  DoIndent();
  Print(op->true_case);
  os() << "\n";
  if (!op->true_case.As<ir::Block>()) DecIndent();

  DoIndent();
  os() << "}";

  if (op->false_case.defined()) {
    os() << " else {\n";

    if (!op->true_case.As<ir::Block>()) IncIndent();
    DoIndent();
    Print(op->false_case);
    os() << "\n";
    if (!op->true_case.As<ir::Block>()) DecIndent();

    DoIndent();
    os() << "}";
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
    CHECK_EQ(op->args.size(), 2UL);
    const ir::_Buffer_ *buffer_arg = op->args.front().As<ir::_Buffer_>();
    CHECK(buffer_arg);

    os() << "cinn_buffer_t* " << buffer_arg->name;
    os() << " = " << op->name;
    os() << "(";
    PrintCastExpr("cinn_device_kind_t", op->args[1]);
    os() << "/*target*/, ";
    PrintRuntimeType(runtime::ToRuntimeType(op->args.front().type().ElementOf()));
    os() << ", ";
    PrintShape(op->args[0].As<ir::_Buffer_>()->shape);
    os() << ")";
  } else if (op->name == runtime::buffer_malloc) {
    CHECK_EQ(op->args.size(), 2UL);
    os() << op->name << "(";
    PrintCastExpr("void*", op->args[0]);
    os() << ", ";
    os() << op->args[1];
    os() << ")";
  } else if (op->call_type == ir::Call::CallType::Intrinsic) {
    CHECK(!op->args.empty());
    os() << op->name << "(";
    for (int i = 0; i < op->args.size() - 1; i++) {
      Print(op->args[i]);
      os() << ", ";
    }
    if (op->args.size() > 0) Print(op->args.back());
    os() << ")";
  } else {
    IrPrinter::Visit(op);
  }
}
void CodeGenC::Visit(const ir::Module *op) { NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::_Var_ *op) { os() << op->name; }
void CodeGenC::Visit(const ir::Load *op) { return ir::IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Store *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Alloc *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Free *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::_Range_ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::_IterVar_ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::_Buffer_ *op) { os() << op->name; }
void CodeGenC::Visit(const ir::_Tensor_ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Let *op) {
  CHECK(op->type().valid());
  os() << PrintType(op->type());
  os() << " ";
  Print(op->value);
  os() << " = ";
  Print(op->body);
}

void CodeGenC::Visit(const ir::Reduce *op) {
  LOG(FATAL) << "Reduce IR is just for internal representation, should not be used for CodeGen.";
}

void CodeGenC::Visit(const ir::Ramp *op) {
  os() << "StackVec<" << op->lanes << "," << PrintType(op->type().ElementOf()) << ">::Ramp(";
  Print(op->base);
  os() << ", ";
  Print(op->stride);
  os() << ", ";
  os() << op->lanes;
  os() << ")";
}

void CodeGenC::Visit(const ir::Broadcast *op) {
  os() << "StackVec<" << op->lanes << "," << PrintType(op->type().ElementOf()) << ">::Broadcast(";
  Print(op->value);
  os() << ", ";
  os() << op->lanes << ")";
}

void CodeGenC::PrintCastExpr(const Type &type, Expr e) {
  os() << "(" << PrintType(type) << ")";
  os() << "(";
  Print(e);
  os() << ")";
}
void CodeGenC::PrintCastExpr(const std::string &type, Expr e) {
  os() << "(" << type << ")";
  os() << "(";
  Print(e);
  os() << ")";
}
void CodeGenC::PrintShape(const std::vector<Expr> &shape) {
  os() << "{ ";

  for (int i = 0; i < shape.size() - 1; i++) {
    Print(shape[i]);
    os() << ", ";
  }
  if (shape.size() > 1) Print(shape.back());

  os() << " }";
}

void CodeGenC::Visit(const ir::_LoweredFunc_ *op) {
  os() << "void " << op->name;

  // output arguments
  os() << "(";

  for (int i = 0; i < op->args.size() - 1; i++) {
    PrintFuncArg(op->args[i]);
    os() << ", ";
  }
  if (op->args.size() >= 1) {
    PrintFuncArg(op->args.back());
  }

  os() << ")\n";

  DoIndent();
  // os() << "{\n";

  // allocate output buffer
  Expr allocate_output_buffer_expr = ir::Block::Make(op->alloc_output_buffer_exprs);
  Expr buffer_cast_expr            = ir::Block::Make(op->buffer_data_cast_exprs);

  Expr func_body = ir::Block::Make({allocate_output_buffer_expr, buffer_cast_expr, op->body});

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
    auto expr = runtime::BufferCreate(buffer);
    Print(expr);
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

void CodeGenC::GenerateHeaderFile(const lang::Module &module) {
  PrintFileGuardOpen(module.name());
  PrintIncludes();

  for (auto &func : module.functions()) {
    os() << "void " << func->name;
    os() << "(";
    for (int i = 0; i < func->args.size() - 1; i++) {
      PrintFuncArg(func->args[i]);
      os() << ", ";
    }
    if (func->args.size() >= 1) {
      PrintFuncArg(func->args.back());
    }

    os() << ");\n";
    os() << "\n\n";
  }

  PrintFileGuardClose(module.name());
}

void CodeGenC::PrintFuncArg(const ir::Argument &arg) {
  if (arg.is_buffer()) {
    if (arg.is_input()) {
      os() << "const struct cinn_buffer_t *";
    } else {
      os() << "struct cinn_buffer_t *";
    }
  } else if (arg.is_scalar()) {
    os() << PrintType(arg.type()) << " ";
    os() << arg.name();
  } else {
    NOT_IMPLEMENTED
  }
  os() << arg.name();
}

void CodeGenC::PrintRuntimeType(const cinn_type_t &type) {
  if (type == cinn_int32_t()) {
    os() << "cinn_int32_t()";
  } else if (type == cinn_int64_t()) {
    os() << "cinn_int64_t()";
  } else if (type == cinn_float32_t()) {
    os() << "cinn_float32_t()";
  } else if (type == cinn_float64_t()) {
    os() << "cinn_float64_t()";
  } else {
    LOG(FATAL) << "Unknown type is not supported to print";
  }
}

}  // namespace backends
}  // namespace cinn

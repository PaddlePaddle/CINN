#include "cinn/backends/codegen_cuda_dev.h"

#include <fstream>

#include "cinn/optim/remove_nested_block.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace backends {

CodeGenCUDA_Dev::CodeGenCUDA_Dev(Target target) : CodeGenC(target) {}

void CodeGenCUDA_Dev::Compile(const lang::Module &module, const Outputs &outputs) {
  CodeGenC::inline_builtin_codes_ = false;
  if (!outputs.c_header_name.empty()) {
    auto source = CodeGenC::Compile(module, OutputKind::CHeader);
    std::ofstream file(outputs.c_header_name);
    CHECK(file.is_open()) << "failed to open file " << outputs.c_header_name;
    file << source;
    file.close();
    LOG(WARNING) << "Output C header to file " << outputs.c_header_name;
  }

  if (!outputs.cuda_source_name.empty()) {
    auto source = CodeGenC::Compile(module, OutputKind::CImpl);
    std::ofstream file(outputs.cuda_source_name);
    CHECK(file.is_open()) << "failed to open file " << outputs.cuda_source_name;
    file << source;
    file.close();
    LOG(WARNING) << "Output C source to file " << outputs.cuda_source_name;
  }
}

std::string CodeGenCUDA_Dev::Compile(const ir::LoweredFunc &func) {
  Print(Expr(func));
  return ss_.str();
}

void CodeGenCUDA_Dev::Visit(const ir::_LoweredFunc_ *op) {
  os() << "__global__\n";

  PrintFunctionDeclaration(op);
  os() << "\n";

  DoIndent();

  // the allocate_output_buffer_expr is not allowed in cuda device function, so we re-implement func codegen without it
  // here.

  Expr buffer_cast_expr       = ir::Block::Make(op->buffer_data_cast_exprs);
  Expr prepare_arguments_expr = ir::Block::Make(op->argument_prepare_exprs);

  Expr func_body = ir::Block::Make({prepare_arguments_expr, buffer_cast_expr, op->body});

  optim::RemoveNestedBlock(&func_body);

  Print(func_body);
}

}  // namespace backends
}  // namespace cinn

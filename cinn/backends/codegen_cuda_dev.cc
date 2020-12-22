#include "cinn/backends/codegen_cuda_dev.h"

#include <fstream>
#include <set>
#include <unordered_set>

#include "cinn/ir/ir_verify.h"
#include "cinn/optim/remove_nested_block.h"

namespace cinn {
namespace backends {

CodeGenCUDA_Dev::CodeGenCUDA_Dev(Target target) : CodeGenC(target) {}

std::string CodeGenCUDA_Dev::Compile(const ir::Module &module, bool for_nvrtc) {
  for_nvrtc_  = for_nvrtc;
  auto source = Compile(module, OutputKind::CImpl);

  return source;
}

void CodeGenCUDA_Dev::Compile(const ir::Module &module, const Outputs &outputs) {
  ir::IrVerify(Expr(module));

  CodeGenC::inline_builtin_codes_ = false;
  if (!outputs.c_header_name.empty()) {
    auto source = Compile(module, OutputKind::CHeader);
    std::ofstream file(outputs.c_header_name);
    CHECK(file.is_open()) << "failed to open file " << outputs.c_header_name;
    file << source;
    file.close();
    LOG(WARNING) << "Output C header to file " << outputs.c_header_name;
  }

  if (!outputs.cuda_source_name.empty()) {
    auto source = Compile(module, OutputKind::CImpl);
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

std::vector<Expr> CodeGenCUDA_Dev::GenerateBufferAliasExprs(const ir::_LoweredFunc_ *op,
                                                            const std::vector<ir::Buffer> &temp_buffers) {
  std::set<ir::Buffer> temp_buffer_set(temp_buffers.begin(), temp_buffers.end());
  // prepare temp buffer alias
  std::vector<Expr> buffer_alias;
  auto tensors = ir::CollectIRNodes(
      op->body, [&](const Expr *x) { return x->as_tensor() && temp_buffer_set.count(x->as_tensor()->buffer); });

  // unique tensors
  std::set<ir::Tensor> unique_tensors;
  for (auto &e : tensors) {
    unique_tensors.insert(e.as_tensor_ref());
  }

  for (auto &t : unique_tensors) {
    auto data_type     = t->type();
    auto data_ptr_type = data_type;
    data_ptr_type.set_cpp_handle();

    Var t_var(t->name, data_ptr_type);
    Var buf_var(t->buffer->name, data_ptr_type);
    buffer_alias.push_back(ir::Let::Make(t_var, buf_var));
  }

  return buffer_alias;
}

void CodeGenCUDA_Dev::Visit(const ir::_LoweredFunc_ *op) {
  os() << "__global__\n";

  PrintFunctionDeclaration(op);
  os() << "\n";

  DoIndent();

  std::vector<Expr> new_body;

  // auto alloca_temp_buffers = op->CudaPrepareAllocTempBufferExprs();
  auto alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  auto temp_buffer_alias   = GenerateBufferAliasExprs(op, op->temp_bufs);
  auto alis_var_exprs      = op->CudaAliasVarExprs();

#define APPEND_TO_NEW_BODY(field__) new_body.insert(std::end(new_body), std::begin(field__), std::end(field__));
  APPEND_TO_NEW_BODY(alloca_temp_buffers)
  APPEND_TO_NEW_BODY(temp_buffer_alias)
  APPEND_TO_NEW_BODY(alis_var_exprs)

  new_body.push_back(op->body);

  Expr func_body = ir::Block::Make(new_body);

  optim::RemoveNestedBlock(&func_body);

  Print(func_body);
}

void CodeGenCUDA_Dev::Visit(const ir::Alloc *op) {
  CHECK(op->destination.as_buffer());
  PrintTempBufferCreation(op->destination.as_buffer_ref());
}

void CodeGenCUDA_Dev::Visit(const ir::Min *op) {
  os() << "cinn_nvgpu_min_fp32(";
  Print(op->a());
  os() << ", ";
  Print(op->b());
  os() << ")";
}

void CodeGenCUDA_Dev::Visit(const ir::Max *op) {
  os() << "cinn_nvgpu_max_fp32(";
  Print(op->a());
  os() << ", ";
  Print(op->b());
  os() << ")";
}

void CodeGenCUDA_Dev::PrintFunctionDeclaration(const ir::_LoweredFunc_ *op) {
  // os() << "void " << GenKernelName(op->name) << "(";
  os() << "void " << op->name << "(";
  for (int i = 0; i < op->args.size() - 1; i++) {
    auto &arg = op->args[i];
    PrintFuncArg(arg);
    os() << ", ";
  }
  if (!op->args.empty()) {
    PrintFuncArg(op->args.back());
  }
  os() << ")";
}

void CodeGenCUDA_Dev::PrintFuncArg(const ir::Argument &arg) {
  if (arg.is_buffer()) {
    // In CUDA kernel, only primitive type is supported, so we replace the buffer with T*j
    if (arg.is_input()) os() << "const ";
    os() << GetTypeRepr(arg.buffer_arg()->dtype);
    os() << "* ";
    os() << kCKeywordRestrict << " ";
    os() << ir::BufferGetTensorName(arg.buffer_arg().As<ir::_Buffer_>());
  } else if (arg.is_var()) {
    if (arg.var_arg()->type().is_cpp_handle()) {
      os() << kCKeywordRestrict;
    }
    os() << GetTypeRepr(arg.type()) << " ";
    os() << arg.name();
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void CodeGenCUDA_Dev::PrintBuiltinCodes() {
  os() << R"ROC(
)ROC";
}

std::string CodeGenCUDA_Dev::Compile(const ir::Module &module, CodeGenC::OutputKind output_kind) {
  ss_.str("");

  if (for_nvrtc_) {
    os() << "extern \"C\" {\n\n";
  }

  if (output_kind == OutputKind::CHeader) {
    GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    PrintIncludes();

    PrintBuiltinCodes();

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    LOG(FATAL) << "Not supported OutputKind";
  }

  if (for_nvrtc_) {
    os() << "\n\n}";
  }

  return ss_.str();
}

void CodeGenCUDA_Dev::PrintIncludes() {
  os() << "#include \"cinn_cuda_runtime_source.cuh\"\n\n";
  os() << "#ifdef __CUDACC_RTC__\n";
  os() << "typedef int int32_t;\n";
  os() << "typedef char int8_t;\n";
  os() << "#endif\n";
  os() << "\n";
  os() << "\n";
}

void CodeGenCUDA_Dev::PrintTempBufferCreation(const ir::Buffer &buffer) {
  CHECK_NE(buffer->type(), Void());
  auto print_gpu_memory = [&](const std::string &mark) {
    os() << mark << GetTypeRepr(buffer->dtype) << " " << buffer->name << " ";

    os() << "[ ";
    for (int i = 0; i < buffer->shape.size() - 1; i++) {
      Print(buffer->shape[i]);
      os() << " * ";
    }
    if (!buffer->shape.empty()) {
      Print(buffer->shape.back());
    }
    os() << " ]";
  };
  switch (buffer->memory_type) {
    case ir::MemoryType::GPUShared:
      print_gpu_memory("__shared__ ");
      break;

    case ir::MemoryType::GPULocal:
      print_gpu_memory("");
      break;

    default:
      LOG(FATAL) << "CUDA device codegen not support memory type " << buffer->memory_type;
  }
}

void CodeGenCUDA_Dev::Visit(const ir::Call *op) {
  os() << op->name + "(";
  Print(op->read_args);
  os() << ")";
}

}  // namespace backends
}  // namespace cinn

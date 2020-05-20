#include "cinn/backends/extern_func_emitter_builtin.h"

#include <glog/logging.h>

#include "cinn/backends/llvm/ir_builder_mixin.h"
#include "cinn/backends/llvm/llvm_util.h"

namespace cinn {
namespace backends {

// C tanh --
// @{
const char* ExternFuncEmitter_C_tanh::func_name() const { return extern_func__tanh; }
void ExternFuncEmitter_C_tanh::EmitImpl(const ir::Call* op) {
  CHECK(codegen_) << "codegen_ should be bind first";

  CHECK_EQ(op->read_args.size(), 1UL);
  CHECK(op->write_args.empty());

  codegen_->os() << extern_tanh_host_repr << "(";
  codegen_->Print(op->read_args[0]);
  codegen_->os() << ")";
}
bool ExternFuncEmitter_C_tanh::RetValuePacked() const { return false; }
const char* ExternFuncEmitter_C_tanh::backend_kind() const { return backend_C; }
void ExternFuncEmitter_C_tanh::BindCodeGen(void* codegen) {
  CHECK(codegen);
  codegen_ = reinterpret_cast<CodeGenC*>(codegen);
}
// @}

// LLVM tanh --
// @{
void ExternFuncEmitter_LLVM_tanh::BindCodeGen(void* codegen) { codegen_ = reinterpret_cast<CodeGenLLVM*>(codegen); }
const char* ExternFuncEmitter_LLVM_tanh::func_name() const { return extern_func__tanh; }
void ExternFuncEmitter_LLVM_tanh::EmitImpl(const ir::Call* op) {
  CHECK(codegen_);
  CodeGenLLVMforEmitter codegen_for_emitter(codegen_);

  // function type.
  llvm::Type* f32             = codegen_for_emitter.b()->getFloatTy();
  llvm::FunctionType* fn_type = llvm::FunctionType::get(f32, {f32}, false);

  llvm::Function* custom_function = llvm::dyn_cast<llvm::Function>(
      codegen_for_emitter.m()->getOrInsertFunction(extern_tanh_host_repr, fn_type).getCallee());
  custom_function->setCallingConv(llvm::CallingConv::C);

  auto* arg = codegen_->Visit(&op->read_args[0]);

  auto* ret = codegen_for_emitter.b()->CreateCall(custom_function, {arg});

  codegen_->extern_func_emit_res_ = ret;
}

bool ExternFuncEmitter_LLVM_tanh::RetValuePacked() const { return false; }
const char* ExternFuncEmitter_LLVM_tanh::backend_kind() const { return backend_llvm_host; }
// @}

// @{
void ExternFuncEmitter_C_tanh_v::BindCodeGen(void* codegen) { codegen_ = reinterpret_cast<CodeGenC*>(codegen); }
const char* ExternFuncEmitter_C_tanh_v::func_name() const { return extern_func__tanh_v; }
void ExternFuncEmitter_C_tanh_v::EmitImpl(const ir::Call* op) {
  auto& os = codegen_->os();
  os << extern_tanh_v_host_repr;
  os << "(";
  codegen_->Print(op->read_args[0]);
  os << ", ";
  codegen_->Print(op->write_args[0]);
  os << ")";
}
bool ExternFuncEmitter_C_tanh_v::RetValuePacked() const { return true; }
const char* ExternFuncEmitter_C_tanh_v::backend_kind() const { return backend_C; }
// @}

// @{
void ExternFuncEmitter_LLVM_tanh_v::BindCodeGen(void* codegen) { codegen_ = reinterpret_cast<CodeGenLLVM*>(codegen); }
const char* ExternFuncEmitter_LLVM_tanh_v::func_name() const { return extern_func__tanh_v; }
void ExternFuncEmitter_LLVM_tanh_v::EmitImpl(const ir::Call* op) {
  CHECK(codegen_);
  CodeGenLLVMforEmitter codegen_for_emitter(codegen_);

  // function type.
  llvm::Type* buffer_p = backends::llvm_type_of<cinn_buffer_t*>(codegen_->m());
  llvm::Type* void_ty  = codegen_->b()->getVoidTy();

  llvm::FunctionType* fn_type = llvm::FunctionType::get(void_ty, {buffer_p, buffer_p}, false);

  llvm::Function* custom_function = llvm::dyn_cast<llvm::Function>(
      codegen_for_emitter.m()->getOrInsertFunction(extern_tanh_v_host_repr, fn_type).getCallee());
  CHECK(custom_function) << "No function called " << extern_tanh_v_host_repr;
  custom_function->setCallingConv(llvm::CallingConv::C);

  auto* arg  = codegen_for_emitter.GetVar(op->read_args[0].as_tensor()->buffer->name, false /*lazy*/);
  auto* arg1 = codegen_for_emitter.GetVar(op->write_args[0].as_tensor()->buffer->name, false /*lazy*/);

  auto* ret = codegen_for_emitter.b()->CreateCall(custom_function, {arg, arg1});

  codegen_->extern_func_emit_res_ = ret;
}
bool ExternFuncEmitter_LLVM_tanh_v::RetValuePacked() const { return true; }
const char* ExternFuncEmitter_LLVM_tanh_v::backend_kind() const { return backend_llvm_host; }
// @}

}  // namespace backends
}  // namespace cinn

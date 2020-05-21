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

void ExternFunctionLLVMEmitter::BindCodeGen(void* codegen) { codegen_ = reinterpret_cast<CodeGenLLVM*>(codegen); }

const char* ExternFunctionLLVMEmitter::func_name() const { return fn_name_.c_str(); }

bool ExternFunctionLLVMEmitter::RetValuePacked() const { return fn_proto().ret_type.is_void(); }

FunctionProto& ExternFunctionLLVMEmitter::fn_proto() const {
  auto* proto = ExternFunctionProtoRegistry::Global().Lookup(fn_name_);
  CHECK(proto) << "No function prototype found for " << fn_name_;
  return *proto;
}
llvm::Type* ExternFunctionLLVMEmitter::llvm_fn_type() const {
  auto* proto = ExternFunctionProtoRegistry::Global().Lookup(fn_name_);
  CHECK(proto) << "No function prototype found for " << fn_name_;

  auto* llvm_ret_type = CinnTypeToLLVMType(proto->ret_type, codegen_->m());
  std::vector<llvm::Type*> arg_types;
  for (auto& t : proto->readonly_arg_types) {
    arg_types.push_back(CinnTypeToLLVMType(t, codegen_->m()));
  }
  for (auto& t : proto->mutable_arg_types) {
    arg_types.push_back(CinnTypeToLLVMType(t, codegen_->m()));
  }
  return llvm_ret_type;
}
const char* ExternFunctionLLVMEmitter::backend_kind() const { return nullptr; }

void ExternFunctionLLVMEmitter::EmitImpl(const ir::Call* op) {
  CHECK(codegen_);
  CodeGenLLVMforEmitter codegen_for_emitter(codegen_);
  llvm::Function* custom_function = llvm::dyn_cast<llvm::Function>(
      codegen_for_emitter.m()->getOrInsertFunction(fn_name_, llvm_fn_type()).getCallee());
  CHECK(custom_function) << "No function registered in JIT called " << fn_name_;
  custom_function->setCallingConv(llvm::CallingConv::C);

  std::vector<llvm::Value*> args;
  for (auto& v : op->read_args) {
    if (v.as_tensor()) {
      args.push_back(codegen_for_emitter.GetVar(v.as_tensor()->buffer->name, false));
    }
  }
  for (auto& v : op->write_args) {
    if (v.as_tensor()) {
      args.push_back(codegen_for_emitter.GetVar(v.as_tensor()->buffer->name, false));
    }
  }

  auto* command                   = codegen_for_emitter.b()->CreateCall(custom_function, args);
  codegen_->extern_func_emit_res_ = command;
}

}  // namespace backends
}  // namespace cinn

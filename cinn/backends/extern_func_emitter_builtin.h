#pragma once

#include <string>
#include <vector>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/extern_func_emitter.h"
#include "cinn/backends/extern_func_protos.h"
#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/backends/llvm/llvm_util.h"

namespace cinn {
namespace backends {

//! Function names

static const char* extern_tanh_host_repr   = "__cinn_host_tanh_fp32";
static const char* extern_tanh_v_host_repr = "__cinn_host_tanh_v";

/**
 * A bridge for the Emitters to access CodeGenLLVM's internal members.
 */
class CodeGenLLVMforEmitter : public CodeGenLLVM {
 public:
  explicit CodeGenLLVMforEmitter(CodeGenLLVM* x) : CodeGenLLVM(x->m(), x->b(), x->named_vars()) {}
};

class ExternFunctionLLVMEmitter : public ExternFunctionEmitter {
 public:
  explicit ExternFunctionLLVMEmitter(const std::string& fn_name) : fn_name_(fn_name) {}

  void BindCodeGen(void* codegen) override;
  const char* func_name() const override;
  bool RetValuePacked() const override;
  const char* backend_kind() const override;

 protected:
  void EmitImpl(const ir::Call* op) override;
  FunctionProto& fn_proto() const;
  llvm::FunctionType* llvm_fn_type() const;

  CodeGenLLVM* codegen_{};
  std::string fn_name_;
};

}  // namespace backends
}  // namespace cinn

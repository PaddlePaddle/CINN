#pragma once

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cinn/backends/llvm/ir_builder_mixin.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/module.h"

namespace cinn {
namespace backends {

class LLVMIRVisitor : public ir::IRVisitorBase<llvm::Value *> {
 public:
  LLVMIRVisitor() = default;

  using ir::IRVisitorBase<llvm::Value *>::Visit;
#define __m(t__) virtual llvm::Value *Visit(const ir::t__ *x) = 0;
  NODETY_FORALL(__m)
#undef __m
};

class CodeGenLLVM : public LLVMIRVisitor, public IrBuilderMixin<CodeGenLLVM> {
 public:
  explicit CodeGenLLVM(llvm::Module *m,
                       llvm::IRBuilder<> *b,
                       std::shared_ptr<std::unordered_map<std::string, llvm::Value *>> vars = nullptr);
  virtual ~CodeGenLLVM();

  llvm::Module *m() { return m_; }
  llvm::IRBuilder<> *b() { return b_; }

  void Compile(const lang::Module &module);

  using LLVMIRVisitor::Visit;

#define __(op__) llvm::Value *Visit(const ir::op__ *) override;
  NODETY_FORALL(__)
#undef __

  //! Used for the ExternFuncEmitter to store temporary result.
  mutable llvm::Value *extern_func_emit_res_{};

  std::shared_ptr<std::unordered_map<std::string, llvm::Value *>> named_vars() { return named_vars_; }

 protected:
  llvm::Value *GetVar(const std::string &name, bool lazy = true);

  // TODO(Superjomn) When to clear the existing local variables when switch to another function?
  llvm::Value *SetVar(const std::string &name, llvm::Value *val);

  //! Visit different kinds of Calls, the following methods are analogous to
  //! those in CodeGenC.
  // @{
  llvm::Value *EmitCall_buffer_create(const ir::Call *op);
  llvm::Value *EmitCall_buffer_malloc(const ir::Call *op);
  llvm::Value *EmitCall_get_address(const ir::Call *op);
  llvm::Value *EmitCall_debug_info(const ir::Call *op);
  // @}

  llvm::Value *EmitBinaryOp(llvm::Value *lhs, llvm::Value *rhs, char opcode, bool is_integral, bool is_signed = true);

  llvm::Value *EmitIntegerUnaryOp(llvm::Value *);
  llvm::Value *EmitFloatUnaryOp(llvm::Value *, llvm::Value *);

  llvm::Value *EmitIntegerBinaryOp(llvm::Value *);
  llvm::Value *EmitFloatBinaryOp(llvm::Value *, llvm::Value *);

  llvm::Value *LLVMGenGlobalStringVar(const std::string &data);

  llvm::Module *m_;
  llvm::IRBuilder<> *b_;

  std::shared_ptr<std::unordered_map<std::string, llvm::Value *>> named_vars_;
};

}  // namespace backends
}  // namespace cinn

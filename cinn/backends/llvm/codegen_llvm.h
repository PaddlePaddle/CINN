#pragma once

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

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
  explicit CodeGenLLVM(llvm::Module *m, llvm::IRBuilder<> *b);
  virtual ~CodeGenLLVM();

  llvm::IRBuilder<> *b() { return b_; }
  llvm::Module *m() { return m_; }

  void Compile(const lang::Module &module);

  using LLVMIRVisitor::Visit;

#define __LLVM_IR_EMITTER_OVERRIDE_VISIT(__op) llvm::Value *Visit(const ::cinn::ir::__op *) override

  __LLVM_IR_EMITTER_OVERRIDE_VISIT(IntImm);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(UIntImm);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(FloatImm);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Add);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Sub);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Mul);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Div);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Mod);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(EQ);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(NE);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(LT);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(LE);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(GT);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(GE);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(And);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Or);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Min);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Max);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Minus);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Not);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Cast);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(For);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(PolyFor);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Select);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(IfThenElse);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Block);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Call);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(_Module_);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(_Var_);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Load);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Store);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Alloc);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Free);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(_Range_);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(_IterVar_);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(_Buffer_);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(_Tensor_);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(_LoweredFunc_);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Let);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Reduce);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Ramp);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Broadcast);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(FracOp);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Power);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Product);
  __LLVM_IR_EMITTER_OVERRIDE_VISIT(Sum);

#undef __LLVM_IR_EMITTER_OVERRIDE_VISIT

 protected:
  llvm::Value *EmitBinaryOp(llvm::Value *lhs, llvm::Value *rhs, char opcode, bool is_integral, bool is_signed = true);

  llvm::Value *EmitIntegerUnaryOp(llvm::Value *);
  llvm::Value *EmitFloatUnaryOp(llvm::Value *, llvm::Value *);

  llvm::Value *EmitIntegerBinaryOp(llvm::Value *);
  llvm::Value *EmitFloatBinaryOp(llvm::Value *, llvm::Value *);

  llvm::Module *m_;
  llvm::IRBuilder<> *b_;

  std::unordered_map<std::string, llvm::Value *> named_vars_;
};

}  // namespace backends
}  // namespace cinn

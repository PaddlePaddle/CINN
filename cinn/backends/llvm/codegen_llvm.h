#pragma once

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cinn/backends/llvm/ir_builder_mixin.h"
#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/ir/intrinsic_ops.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/ir/module.h"

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

/**
 * Tell whether a variable called \p \var_name will lowered to a pointer type in LLVM.
 * @param var_name name of the variable.
 * @return a boolean.
 */
bool LLVM_WillVarLowerAsPointer(const std::string &var_name);

class SymbolTable {
 public:
  SymbolTable() = default;

  void PushScope() { scopes_.emplace_back(); }

  llvm::Value *Lookup(const std::string &id) {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); it++) {
      auto vt = (*it).find(id);
      if (vt != (*it).end()) return vt->second;
    }
    return nullptr;
  }

  void Insert(const std::string &id, llvm::Value *value) {
    CHECK(!scopes_.empty());
    scopes_.back().emplace(id, value);
  }

  void Erase(const std::string &id) {
    CHECK(!scopes_.empty());
    scopes_.back().erase(id);
  }

  void PopScope() {
    CHECK(!scopes_.empty());
    scopes_.pop_back();
  }

  //! Get the number of the variables contained in the current scope.
  size_t size() const { return scopes_.empty() ? 0 : scopes_.back().size(); }

  size_t num_scopes() const { return scopes_.size(); }

 private:
  std::vector<std::unordered_map<std::string, llvm::Value *>> scopes_;

  SymbolTable(const SymbolTable &) = delete;
};

struct SymbolTableGuard {
  explicit SymbolTableGuard(SymbolTable &symbol_table) : symbol_table_(symbol_table) { symbol_table.PushScope(); }

  ~SymbolTableGuard() { symbol_table_.PopScope(); }

 private:
  SymbolTable &symbol_table_;
};

/**
 * Base class of all the LLVM-based codegen.
 */
class CodeGenLLVM : public LLVMIRVisitor, public IrBuilderMixin<CodeGenLLVM> {
 public:
  explicit CodeGenLLVM(llvm::Module *m,
                       llvm::IRBuilder<> *b,
                       const std::shared_ptr<SymbolTable> &symbol_table = nullptr,
                       const Target &target                             = common::DefaultHostTarget());

  // Common llvm types
  // @{
  inline llvm::Type *ll_void_p_ty() const { return llvm_type_of<void *>(m_); }
  inline llvm::Type *ll_void_pp_ty() const { return llvm_type_of<void **>(m_); }
  inline llvm::Type *ll_int8_ty() const { return llvm_type_of<int8_t>(m_); }
  inline llvm::Type *ll_int32_ty() const { return llvm_type_of<int32_t>(m_); }
  inline llvm::Type *ll_fp32_ty() const { return llvm_type_of<float>(m_); }
  inline llvm::Type *ll_fp64_ty() const { return llvm_type_of<double>(m_); }
  inline llvm::Type *ll_cinn_buffer_p_ty() const { return llvm_type_of<cinn_buffer_t *>(m_); }
  inline llvm::Type *ll_cinn_pod_ty() const { return llvm_type_of<cinn_pod_value_t>(m_); }
  inline llvm::Type *ll_cinn_pod_p_ty() const { return llvm_type_of<cinn_pod_value_t *>(m_); }
  // @}

  //! get a llvm type equivalent to a CINN type.
  inline llvm::Type *ll_type_of(Type type) { return CinnTypeToLLVMType(type, m_); }

  // Common methods to get a constant
  // @{
  inline llvm::Constant *ll_const_int32(int v) const { return llvm::ConstantInt::get(b_->getInt32Ty(), v); }
  // @}

  //! Get the bound LLVM module.
  llvm::Module *m() { return m_; }
  //! Get the bound LLVM ir builder.
  llvm::IRBuilder<> *b() { return b_; }

  void Compile(const ir::Module &module);

  using LLVMIRVisitor::Visit;

#define __(op__) llvm::Value *Visit(const ir::op__ *) override;
  NODETY_FORALL(__)
#undef __

#define __(op__) llvm::Value *Visit(const ir::intrinsics::op__ *);
  INTRINSIC_KIND_FOR_EACH(__)
#undef __

  //! Used for the ExternFuncEmitter to store temporary result.
  mutable llvm::Value *extern_func_emit_res_{};

  std::shared_ptr<SymbolTable> named_vars() { return symbol_table_; }

  llvm::FunctionType *GenFunctionTypeFromCinnFunction(const ir::_LoweredFunc_ *func, bool with_buffer_type);

  virtual llvm::Value *GetVar(const std::string &name, bool lazy = true);

  llvm::Function *GetIntrinsicDecl(llvm::Intrinsic::ID id,
                                   llvm::Type *ret_type,
                                   llvm::ArrayRef<llvm::Type *> arg_types);

  // Constants
  // @{
  inline llvm::Value *llvm_int32_constant(int v) { return llvm::ConstantInt::get(ll_int32_ty(), v); }
  // @}

  virtual ~CodeGenLLVM();

 protected:
  // TODO(Superjomn) When to clear the existing local variables when switch to another function?
  llvm::Value *SetVar(const std::string &name, llvm::Value *val);
  llvm::Value *EmitVectorSlice(llvm::Value *vec, int begin, int extent);
  llvm::Value *EmitVectorPad(llvm::Value *vec, int lanes);
  llvm::Value *EmitVectorConcat(std::vector<llvm::Value *> vecs);

  //! Visit different kinds of Calls, the following methods are analogous to
  //! those in CodeGenC.
  // @{
  llvm::Value *EmitCall_buffer_create(const ir::Call *op);
  llvm::Value *EmitCall_buffer_malloc(const ir::Call *op);
  llvm::Value *EmitCall_get_address(const ir::Call *op);
  llvm::Value *EmitCall_debug_info(const ir::Call *op);
  // @}

  llvm::Value *EmitBinaryOp(llvm::Value *lhs, llvm::Value *rhs, char opcode, bool is_integral, bool is_signed = true);

  llvm::Value *LLVMGenGlobalStringVar(const std::string &data);

  llvm::Value *CreateBufferPtr(Type t, llvm::Value *buffer, llvm::Value *index);
  llvm::Value *CreateBufferVecPtr(Type t, llvm::Value *buffer, llvm::Value *index);
  llvm::Value *CreateVecSlice(llvm::Value *vec, int begin, int lanes);

  llvm::Value *DenseVectorLoad(const ir::Load *load);

  /**
   * Mark a load or store with type-based-alias-analysis metadata so that LLVM can optimize by reordering loads and
   * stores accross different buffers.
   */
  void AddTbaaMetadata(llvm::Instruction *inst, std::string_view buffer, Expr index);

  void InitTarget(const Target &target);

  llvm::Module *m_;
  llvm::IRBuilder<> *b_;

  std::unique_ptr<llvm::MDBuilder> md_builder_;

  // std::shared_ptr<std::unordered_map<std::string, llvm::Value *>> named_vars_;
  std::shared_ptr<SymbolTable> symbol_table_;
  std::unordered_set<ir::_Var_ *> alias_vars_;

  llvm::MDNode *md_tbaa_root_{nullptr};
  llvm::MDNode *md_tbaa_alias_set_{nullptr};

  int naive_vec_alignment_{0};
  Target target_;
};
namespace detail {
Expr StridedRampBase(Expr e, int stride);
}  // namespace detail

}  // namespace backends
}  // namespace cinn

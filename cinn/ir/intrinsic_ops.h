#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include "cinn/common/type.h"
#include "cinn/ir/ir.h"

//! This file defines some intrinsic IR nodes, this is similar to the MLIR operations, we try to expose some underlying
//! opaque operations to IR system to helpe more intuitive codegen.

namespace cinn::ir {

enum class IntrinsicKind {
  // All the intrinsics should registered here.
};

class IntrinsicOp : public IrNode {
 public:
  IntrinsicOp(llvm::ArrayRef<Type> input_types, llvm::ArrayRef<Type> output_types)
      : input_types_(input_types.begin(), input_types.end()), output_types_(output_types.begin(), output_types.end()) {}

  const Type& GetInputType(int offset) const;
  const Type& GetOutputType(int offset) const;

  const llvm::SmallVectorImpl<Type>& input_types() const { return input_types_; }
  const llvm::SmallVectorImpl<Type>& output_types() const { return input_types_; }

  //! Verify the \p input_types and \p output_types matches the signature of this operation.
  void Verify(llvm::ArrayRef<Type> input_types, llvm::ArrayRef<Type> output_types);
  void Verify(llvm::ArrayRef<Expr> inputs, llvm::ArrayRef<Expr> outputs);

  void Verify(llvm::ArrayRef<Expr> inputs);

  const char* type_info() const override;
  void Accept(IRVisitor* v) const override;

 protected:
  llvm::SmallVector<Type, 4> input_types_;
  llvm::SmallVector<Type, 4> output_types_;
};

namespace intrinsics {

/**
 * BufferGetMemoryAddr is the operation to get the memory address from cinn_buffer_t.
 */
struct BufferGetMemoryAddr : public IntrinsicOp {
 public:
  // signature: (cinn_buffer_t*) -> (void*)
  BufferGetMemoryAddr() : IntrinsicOp({type_of<cinn_buffer_t*>()}, {type_of<void*>()}) {}

  static Expr Make(Expr buffer);

  Expr buffer;
};

}  // namespace intrinsics

}  // namespace cinn::ir

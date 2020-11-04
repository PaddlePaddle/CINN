#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include "cinn/common/type.h"
#include "cinn/ir/ir.h"

//! This file defines some intrinsic IR nodes, this is similar to the MLIR operations, we try to expose some underlying
//! opaque operations to IR system to helpe more intuitive codegen.

namespace cinn::ir {

// clang-format off
#define INTRINSIC_KIND_FOR_EACH(macro__)                 \
  macro__(BufferGetDataHandle)                           \
  macro__(BufferGetDataConstHandle)                      \
  macro__(PodValueToX)                                   \
  macro__(BufferCreate)                                  \
  macro__(GetAddr)                                       \
  macro__(ArgsConstruct)                                 \
// clang-format on


enum class IntrinsicKind {
  // All the intrinsics should registered here.
#define __(x__) k ## x__,
  INTRINSIC_KIND_FOR_EACH(__)
#undef __
};


class IntrinsicOp : public IrNode {
 public:
  IntrinsicOp(IntrinsicKind kind, llvm::ArrayRef<Type> input_types, llvm::ArrayRef<Type> output_types)
      : kind_(kind),
        input_types_(input_types.begin(), input_types.end()),
        output_types_(output_types.begin(), output_types.end()) {}

  const Type& GetInputType(int offset) const;
  const Type& GetOutputType(int offset) const;

  void AddInputType(const Type& type) { input_types_.push_back(type); }
  void AddOutputType(const Type& type) { output_types_.push_back(type); }

  const llvm::SmallVectorImpl<Type>& input_types() const { return input_types_; }
  const llvm::SmallVectorImpl<Type>& output_types() const { return input_types_; }

  //! Verify the \p input_types and \p output_types matches the signature of this operation.
  void Verify(llvm::ArrayRef<Type> input_types, llvm::ArrayRef<Type> output_types);
  void Verify(llvm::ArrayRef<Expr> inputs, llvm::ArrayRef<Expr> outputs);
  void Verify(llvm::ArrayRef<Expr> inputs);

  const char* type_info() const override;

  IntrinsicKind getKind() const { return kind_; }

  IrNodeTy node_type() const override { return _node_type_; }

  static constexpr IrNodeTy _node_type_{IrNodeTy::IntrinsicOp};

 protected:
  llvm::SmallVector<Type, 4> input_types_;
  llvm::SmallVector<Type, 4> output_types_;
  const IntrinsicKind kind_;
};

namespace intrinsics {

/**
 * The operation to get the memory address from cinn_buffer_t.
 */
struct BufferGetDataHandle : public IntrinsicOp {
  // signature: (cinn_buffer_t*) -> (void*)
  BufferGetDataHandle()
      : IntrinsicOp(IntrinsicKind::kBufferGetDataHandle, {type_of<cinn_buffer_t*>()}, {type_of<void*>()}) {}

  static Expr Make(Expr buffer);

  static bool classof(const IntrinsicOp* s) { return s->getKind() == IntrinsicKind::kBufferGetDataHandle; }

  Expr buffer;
};

/**
 * The operation to get the memory address from cinn_buffer_t.
 */
struct BufferGetDataConstHandle : public IntrinsicOp {
  // signature: (cinn_buffer_t*) -> (const void*)
  BufferGetDataConstHandle()
      : IntrinsicOp(IntrinsicKind::kBufferGetDataConstHandle, {type_of<const cinn_buffer_t*>()}, {type_of<void*>()}) {}

  static Expr Make(Expr buffer);

  static bool classof(const IntrinsicOp* s) { return s->getKind() == IntrinsicKind::kBufferGetDataConstHandle; }

  Expr buffer;
};

/**
 * The operation to represent the helper methods:
 * - cinn_pod_value_to_float
 * - cinn_pod_value_to_duoble
 * - cinn_pod_value_to_int64
 * - cinn_pod_value_to_int32
 * - cinn_pod_value_to_void_p
 * - cinn_pod_value_to_buffer_p
 */
struct PodValueToX : public IntrinsicOp {
  // signature: (cinn_pod_value_t*) -> (X), X is some pod type.
  explicit PodValueToX()
      : IntrinsicOp(IntrinsicKind::kPodValueToX, {type_of<cinn_pod_value_t*>()}, {}) {}

  static Expr Make(Expr pod_value_ptr, const Type& type);

  static bool classof(const IntrinsicOp* s) { return s->getKind() == IntrinsicKind::kPodValueToX; }

  Expr pod_value_ptr;
};

/**
 * The operation to create a buffer.
 */
struct BufferCreate : public IntrinsicOp {
  // signature: (cinn_buffer_t*) -> void
  explicit BufferCreate(): IntrinsicOp(IntrinsicKind::kBufferCreate, {type_of<cinn_buffer_t*>()}, {}) {}

  static Expr Make(Expr buffer);

  static bool classof(const IntrinsicOp* s) { return s->getKind() == IntrinsicKind::kBufferCreate; }

  Expr buffer;
};

/**
 * The operation to get the address of a data.
 */
struct GetAddr : public IntrinsicOp {
  // signature: (X) -> (X*)
  explicit GetAddr(): IntrinsicOp(IntrinsicKind::kGetAddr, {}, {}) {}

  static Expr Make(Expr data);

  static bool classof(const IntrinsicOp* s) { return s->getKind() == IntrinsicKind::kGetAddr; }

  Expr data;
};

/**
 * The operation to construct a cinn_pod_value_t*
 */
struct ArgsConstruct : public IntrinsicOp {
  explicit ArgsConstruct() : IntrinsicOp(IntrinsicKind::kArgsConstruct, {}, {}) {}

  static Expr Make(Var var, llvm::ArrayRef<Expr> args);

  static bool classof(const IntrinsicOp* s) { return s->getKind() == IntrinsicKind::kArgsConstruct; }

  Var var;
  llvm::SmallVector<Expr, 4> args;
};


}  // namespace intrinsics

}  // namespace cinn::ir

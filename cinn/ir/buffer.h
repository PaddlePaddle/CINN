#pragma once

#include <string>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/node.h"

namespace cinn {
namespace ir {

class _Buffer_;
class Tensor;
class _Tensor_;

//! The memory access mode.
enum class AccessMask : int {
  kRead = 1,
  kWrite,
};

/**
 * Buffer is a symbolic multi-dimensional data structure, it is a node in IR.
 * It is a composition of primitive symbolic types, used to specify the memory layout of the Tensor used in the program
 * input. User can create a buffer and bind to multiple Tensors to specify that the tensors are not inlined and persist
 * data to this buffer.
 */
class Buffer : public IrNodeRef {
 public:
  Buffer() = default;
  explicit Buffer(IrNode* n) : IrNodeRef(n) {}

  const _Buffer_* operator->() const;
  _Buffer_* operator->();
};

class _Buffer_ : public ExprNode<_Buffer_> {
 public:
  //! The pointer to the head of the data.
  Var data;
  //! The shape of the buffer.
  std::vector<Expr> shape;
  //! The strides of each dimension.
  // This can be empty, indicating that the array is contiguous.
  std::vector<Expr> strides;
  //! The name of the buffer.
  std::string name;
  //! The storage scope of the buffer, empty if global.
  std::string scope;
  //! Aignment requirement of data pointer in bytes.
  int data_alignment;
  //! The offset in terms of number of dtype elements (including lanes).
  Expr elem_offset;
  //! Factor of elem_offset field.
  // elem_offset is guaranteed to be multiple of offset_factor.
  int offset_factor;

  _Buffer_() = default;

  static Buffer Make(Var data,
                     Type dtype,
                     const std::vector<Expr>& shape,
                     const std::vector<Expr>& strides,
                     Expr elem_offset,
                     const std::string& name,
                     const std::string& scope,
                     int data_alignment,
                     int offset_factor);

  static Buffer Make(const std::string& name, const std::vector<Expr>& shape = {});

  //! Make an empty buffer.
  static Buffer Make();

  void BindTo(const Tensor& tensor);
  void BindTo(const _Tensor_* tensor);

  void Accept(IRVisitor* v) const override;
  IrNodeTy node_type() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_Buffer_;

 private:
  std::set<std::string> bound_tensors_names_;
};

}  // namespace ir
}  // namespace cinn

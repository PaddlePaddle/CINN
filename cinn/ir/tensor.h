#pragma once

#include "cinn/ir/function_base.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

class _Tensor_;
/**
 * Tensor representing a possible input or intermediate computation result.
 */
class Tensor : public IrNodeRef {
 public:
  Tensor() = default;
  explicit Tensor(IrNode* n) : IrNodeRef(n) {}

  inline const _Tensor_* operator->() const;

  //! \return The dimension of the tensor.
  inline size_t ndims() const;

  /**
   * Take elements from the tensor.
   * @param args The indices.
   * @return The result expression representing a tensor read.
   */
  template <typename... Args>
  inline Expr operator()(Args&&... args) const {
    std::vector<Expr> indices(std::forward<Args>(args)...);
    return operator()(indices);
  }

  /**
   * Take elements from the tensor.
   * @param indices The indices.
   * @return The result expression representing a tensor read.
   */
  Expr operator()(const std::vector<Expr>& indices);

  /**
   * Take elements from the tensor.
   * @param indices The indices.
   * @return The result expression representing a tensor read.
   */
  Expr operator()(const std::vector<Var>& indices);

  /**
   * Data structure to represent a slice that fixes first k coordinates.
   */
  class Slice {
   public:
    Slice(const Tensor& tensor, const std::vector<Expr>& indices) : tensor_(tensor), indices_(indices) {}

    /**
     * Get i-th slice from the current slice.
     * @param i the indice of the coordinate.
     * @return the subsequent slice.
     */
    inline Slice operator[](Expr i) {
      std::vector<Expr> other = indices_;
      other.emplace_back(i);
      return Slice(tensor_, other);
    }
    /**
     * Convert slice to expression.
     * @return The corresponding expression of this slice.
     */
    inline operator Expr() const { return tensor_(indices_); }

   private:
    const Tensor& tensor_;
    std::vector<Expr> indices_;
  };
};

class _Operation_;
/**
 * Operation that produces tensors.
 */
class Operation : public FunctionRef {
 public:
  Operation() = default;
  explicit Operation(IrNode* n) : FunctionRef(n) {}

  inline const _Operation_* operator->() const;

  //! Get the i-th output of the operation.
  Tensor output(size_t i) const;
};

class _Tensor_ : public IrNode {
 public:
  //! The shape of the tensor.
  std::vector<Expr> shape;
  //! The data type of the elements in the tensor.
  Type dtype;
  //! The source operation, can be None.
  Operation op;
  //! The output index from source operation.
  size_t value_index{};

  _Tensor_() = default;

  static Tensor Make(const std::vector<Expr>& shape, Type dtype, Operation op, int value_index);
  void Accept(IrVisitor* v) const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_Tensor_;
};

}  // namespace ir
}  // namespace cinn

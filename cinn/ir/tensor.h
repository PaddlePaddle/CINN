#pragma once

#include <map>
#include "cinn/common/graph_utils.h"
#include "cinn/ir/function_base.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

class _Tensor_;

namespace detail {
constexpr bool LE(int a, int b) { return a <= b; }
constexpr bool GE(int a, int b) { return a >= b; }
}  // namespace detail
   /**
    * Tensor representing a possible input or intermediate computation result.
    */
class Tensor : public IrNodeRef, common::GraphNode {
 public:
  Tensor() = default;
  explicit Tensor(IrNode* n) : IrNodeRef(n) {}
  Tensor(const std::vector<Var>& shape, Type type = Float(32));
  Tensor(const std::vector<Expr>& shape, Type type = Float(32));

  inline const _Tensor_* operator->() const;
  inline _Tensor_* operator->();

  //! \return The dimension of the tensor.
  inline size_t ndims() const;

  /**
   * Take elements from the tensor.
   * @param args The indices.
   * @return The result expression representing a tensor read.
   */
  template <typename... Args>
  inline typename std::enable_if<detail::GE(sizeof...(Args), 2), Expr>::type operator()(Args&&... args) const {
    std::vector<Expr> indices({std::forward<Args>(args)...});
    return operator()(indices);
  }

  /**
   * Take elements from the tensor.
   * @param indices The indices.
   * @return The result expression representing a tensor read.
   */
  Expr operator()(const std::vector<Expr>& indices) const;

  /**
   * Take elements from the tensor.
   * @param indices The indices.
   * @return The result expression representing a tensor read.
   */
  Expr operator()(const std::vector<Var>& indices) const;

  inline bool operator==(const Tensor& other) const;

  IrNodeTy node_type() const override;

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

  std::string name;
};

/*
class _Tensor_ : public IrNode {
 public:
  //! The shape of the tensor.
  std::vector<Expr> shape;
  //! The source operation, can be None.
  Operation op;
  //! The output index from source operation.
  size_t value_index{};

  _Tensor_() = default;

  static Tensor Make(const std::vector<Expr>& shape, Type dtype, Operation op, int value_index);
  void Accept(IrVisitor* v) const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_Tensor_;
};
 */

class _Operation_ : public ir::FunctionBase {
 public:
  //! Optional name of the operation.
  std::string name;
  //! Optional tag of the operation.
  std::string tag;
  //! Additional attributes of the operation.
  std::map<std::string, IrNodeRef> attrs;

  const std::string& func_name() const final { return name; }
};

}  // namespace ir
}  // namespace cinn

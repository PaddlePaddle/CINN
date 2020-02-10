#pragma once

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/node.h"
#include "cinn/poly/element.h"

namespace cinn {
namespace lang {

using ir::Expr;
using ir::IrNodeRef;
using ir::Type;
using ir::Var;

namespace detail {
constexpr bool LE(int a, int b) { return a <= b; }
constexpr bool GE(int a, int b) { return a >= b; }
}  // namespace detail

/**
 * @brief Tensor representing a possible input, or intermediate computation result.
 */
class Tensor : public ir::IrNodeRef {
 public:
  Tensor() = default;
  explicit Tensor(ir::IrNode* n) : IrNodeRef(n) {}
  Tensor(const std::vector<Expr>& shape, const std::vector<Var>& iterators, Type dtype, ir::Expr expr);

  //! Get number of dimensions.
  inline size_t ndims() const;

  inline const ir::_Tensor_* operator->() const { return As<ir::_Tensor_>(); }
  inline ir::_Tensor_* operator->() { return As<ir::_Tensor_>(); }

  /**
   * Take elements from the tensor.
   * This take one or multiple expression as indices.
   */
  // @{
  Expr operator()(const Expr& a) const { return operator()({a}); }
  template <typename... Args>
  inline typename std::enable_if<detail::GE(sizeof...(Args), 2), Expr>::type operator()(Args... args) const {
    return operator()({std::forward<Args>(args)...});
  }
  // @}

  /**
   * Take elements from the tensor.
   * @param indices  The indices.
   * @return The result expression representing a tensor read.
   */
  Expr operator()(const std::vector<Expr>& indices) const;
};

}  // namespace lang
}  // namespace cinn

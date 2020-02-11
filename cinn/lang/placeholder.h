#pragma once
#include <cinn/ir/operation.h>

#include "cinn/common/common.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace lang {

using ir::Expr;

/**
 * Placeholder
 * @tparam T
 */
template <typename T>
class Placeholder {
 public:
  Placeholder(const std::string &name, const std::vector<Expr> &shape);

  //! Get a slice.
  // @{
  Expr operator()(Expr a) const { return operator()({a}); }
  Expr operator()(Expr a, Expr b) const { return operator()({a, b}); }
  Expr operator()(Expr a, Expr b, Expr c) const { return operator()({a, b, c}); }
  Expr operator()(Expr a, Expr b, Expr c, Expr d) const { return operator()({a, b, c, d}); }

  Expr operator()(ir::Var a, ir::Var b) { return operator()(Expr(a), Expr(b)); }
  Expr operator()(const std::vector<Expr> &indices) const;
  // @}

 private:
  ir::Tensor tensor_;
};

template <typename T>
Expr Placeholder<T>::operator()(const std::vector<Expr> &indices) const {
  return tensor_(indices);
}

template <typename T>
Placeholder<T>::Placeholder(const std::string &name, const std::vector<Expr> &shape) {
  ir::Var buffer_ptr(Context::Global().NewName("buffer"));
  buffer_ptr->set_type(type_of<T>());

  std::vector<Expr> strides(shape.size(), Expr(1));
  Expr offset(0);

  std::vector<ir::Var> axis;
  for (int i = 0; i < shape.size(); i++) axis.emplace_back(common::axis_name(i));

  auto op = ir::PlaceholderOp::Make(name, shape, type_of<T>());

  tensor_ = ir::_Tensor_::Make(name, shape, op);
}

}  // namespace lang
}  // namespace cinn

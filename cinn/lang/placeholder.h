#pragma once
#include "cinn/common/common.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
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
  Placeholder(const std::vector<Expr>& shape) {
    ir::Var buffer_ptr(Context::Global().NewName("buffer"));
    std::vector<Expr> strides(shape.size(), Expr(1));
    Expr offset(0);
    buffer_ = ir::_Buffer_::Make(buffer_ptr,
                                 type_of<T>(),  // data type
                                 shape,         //
                                 strides,
                                 offset,
                                 buffer_ptr->name,
                                 "",
                                 0,
                                 0);
    LOG(INFO) << buffer_->node_type();
  }

  //! Get a slice.
  // @{
  Expr operator()(Expr a) const { return operator()({a}); }
  Expr operator()(Expr a, Expr b) const { return operator()({a, b}); }
  Expr operator()(Expr a, Expr b, Expr c) const { return operator()({a, b, c}); }
  Expr operator()(Expr a, Expr b, Expr c, Expr d) const { return operator()({a, b, c, d}); }
  Expr operator()(const std::vector<Expr>& indice) const;
  // @}

 private:
  ir::Buffer buffer_;
};

template <typename T>
Expr Placeholder<T>::operator()(const std::vector<Expr>& indice) const {
  return runtime::BufferLoad(buffer_, indice);
}

}  // namespace lang
}  // namespace cinn
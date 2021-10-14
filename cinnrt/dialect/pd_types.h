// This file defines the types used in PaddlePaddle MLIR dialect.
// We borrowed much ideas from tensorflow mlir dialect (tf_types.h in tensorflow).

#pragma once

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace pd {

class PaddleType : public Type {
 public:
  using Type::Type;

  static bool classof(Type type);
};

namespace detail {

template <typename Derived>
class PaddleTypeImpl : public Type::TypeBase<Derived, PaddleType, TypeStorage> {
 public:
  using Base   = typename Type::TypeBase<Derived, PaddleType, TypeStorage>;
  using PDBase = PaddleTypeImpl<Derived>;
  using Base::Base;
};

}  // namespace detail

#define HANDLE_PD_TYPE(pdtype, enumerant, name)                      \
  class pdtype##Type : public detail::PaddleTypeImpl<pdtype##Type> { \
   public:                                                           \
    using PDBase::PDBase;                                            \
  };

}  // namespace pd
}  // namespace mlir

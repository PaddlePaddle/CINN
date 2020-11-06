#pragma once
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;  // NOLINT
namespace cinn::dt {

class TensorType : public mlir::Type::TypeBase<TensorType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

#include "cinnrt/dialect/dense_tensor_dialect.hpp.inc"

#define GET_OP_CLASSES
#include "cinnrt/dialect/dense_tensor.hpp.inc"

}  // namespace cinn::dt

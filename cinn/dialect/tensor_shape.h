#pragma once
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;  // NOLINT
namespace cinn::ts {

class ShapeType : public Type::TypeBase<ShapeType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class PartialShapeType : public Type::TypeBase<PartialShapeType, Type, TypeStorage> {
 public:
  using Base::Base;
};

#define GET_OP_CLASSES
#include "cinn/dialect/tensor_shape.hpp.inc"
#include "cinn/dialect/tensor_shape_dialect.hpp.inc"

}  // namespace cinn::ts

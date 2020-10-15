#pragma once
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace cinn::dt {
using namespace mlir;  // NOLINT

#define GET_OP_CLASSES
#include "cinn/dialect/dense_tensor.hpp.inc"
#undef GET_OP_CLASSES

#include "cinn/dialect/dense_tensor_dialect.hpp.inc"

}  // namespace cinn::dt

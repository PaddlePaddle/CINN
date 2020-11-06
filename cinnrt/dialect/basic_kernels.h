#pragma once
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace cinn::dialect {
using namespace mlir;  // NOLINT
#define GET_OP_CLASSES
#include "cinn/dialect/basic_kernels.hpp.inc"
}  // namespace cinn::dialect

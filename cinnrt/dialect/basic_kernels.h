#pragma once
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;  // NOLINT

namespace cinnrt::dialect {
#define GET_OP_CLASSES
#include "cinnrt/dialect/basic_kernels.hpp.inc"
}  // namespace cinnrt::dialect

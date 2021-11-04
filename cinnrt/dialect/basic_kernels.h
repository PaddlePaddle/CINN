#pragma once
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;  // NOLINT

namespace infrt::dialect {
#define GET_OP_CLASSES
#include "infrt/dialect/basic_kernels.hpp.inc"
}  // namespace infrt::dialect

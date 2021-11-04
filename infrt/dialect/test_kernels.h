#pragma once
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace infrt::dialect {
using namespace mlir;  // NOLINT
#define GET_OP_CLASSES
#include "infrt/dialect/test_kernels.hpp.inc"
}  // namespace infrt::dialect

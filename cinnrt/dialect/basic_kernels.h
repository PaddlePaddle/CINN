#pragma once
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace cinnrt::dialect {
using namespace mlir;  // NOLINT
#define GET_OP_CLASSES
#include "cinnrt/dialect/basic_kernels.hpp.inc"
}  // namespace cinnrt::dialect

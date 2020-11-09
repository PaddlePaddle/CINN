#pragma once

#include "mlir/IR/Dialect.h"

namespace cinnrt {

void RegisterCinnDialects(mlir::DialectRegistry& registry);

}  // namespace cinnrt

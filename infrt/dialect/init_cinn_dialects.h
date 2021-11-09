#pragma once

#include "mlir/IR/Dialect.h"

namespace infrt {

void RegisterCinnDialects(mlir::DialectRegistry& registry);

}  // namespace infrt

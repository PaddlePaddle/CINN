#pragma once

#include "mlir/IR/Dialect.h"

namespace cinn {

void RegisterCinnDialects(mlir::DialectRegistry& registry);

}  // namespace cinn

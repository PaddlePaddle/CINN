#pragma once

#include "mlir/IR/Dialect.h"

namespace cinn::dialect {

void RegisterCinnDialects(mlir::DialectRegistry& registry);

}  // namespace cinn::dialect

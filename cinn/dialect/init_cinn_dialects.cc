#include "cinn/dialect/init_cinn_dialects.h"

#include "cinn/dialect/basic_kernels.h"
#include "cinn/dialect/cinn_base.h"

namespace cinn::dialect {

void RegisterCinnDialects(mlir::DialectRegistry& registry) { registry.insert<CINN_Dialect>(); }

}  // namespace cinn::dialect

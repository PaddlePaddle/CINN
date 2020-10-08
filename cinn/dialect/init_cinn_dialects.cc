#include "cinn/dialect/init_cinn_dialects.h"

#include "cinn/dialect/basic_kernels.h"
#include "cinn/dialect/cinn_base.h"

namespace cinn::dialect {

void RegisterCinnDialects() { mlir::DialectRegistration<CINN_Dialect>(); }

}  // namespace cinn::dialect

#include "cinn/dialect/init_cinn_dialects.h"

#include <glog/logging.h>
#include "cinn/dialect/basic_kernels.h"
#include "cinn/dialect/cinn_base.h"
#include "cinn/dialect/tensor_shape.h"

namespace cinn {

void RegisterCinnDialects(mlir::DialectRegistry& registry) {
  registry.insert<ts::TensorShapeDialect, dialect::CINNDialect>();
}

}  // namespace cinn

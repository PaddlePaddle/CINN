#include "init_cinn_dialects.h"

#include <glog/logging.h>

#include "basic_kernels.h"
#include "cinn_base.h"
#include "dense_tensor.h"
#include "tensor_shape.h"

namespace cinn {

void RegisterCinnDialects(mlir::DialectRegistry& registry) {
  registry.insert<ts::TensorShapeDialect, dialect::CINNDialect, dt::DTDialect>();
}

}  // namespace cinn

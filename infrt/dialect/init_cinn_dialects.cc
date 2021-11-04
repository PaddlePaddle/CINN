#include "infrt/dialect/init_cinn_dialects.h"

#include <glog/logging.h>

#include "infrt/dialect/basic_kernels.h"
#include "infrt/dialect/cinn_base.h"
#include "infrt/dialect/dense_tensor.h"
#include "infrt/dialect/pd_ops.h"
#include "infrt/dialect/tensor_shape.h"

namespace infrt {

void RegisterCinnDialects(mlir::DialectRegistry& registry) {
  registry.insert<ts::TensorShapeDialect>();
  registry.insert<dialect::CINNDialect>();
  registry.insert<dt::DTDialect>();
  registry.insert<mlir::pd::PaddleDialect>();
}

}  // namespace infrt

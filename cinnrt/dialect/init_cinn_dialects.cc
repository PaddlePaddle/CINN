#include "cinnrt/dialect/init_cinn_dialects.h"

#include <glog/logging.h>

#include "cinnrt/dialect/basic_kernels.h"
#include "cinnrt/dialect/cinn_base.h"
#include "cinnrt/dialect/dense_tensor.h"
#include "cinnrt/dialect/pd_ops.h"
#include "cinnrt/dialect/tensor_shape.h"

namespace cinnrt {

void RegisterCinnDialects(mlir::DialectRegistry& registry) {
  registry.insert<ts::TensorShapeDialect, dialect::CINNDialect, dt::DTDialect, mlir::PD::PaddleDialect>();
}

}  // namespace cinnrt

#include "cinnrt/dialect/pd_ops.h"

namespace mlir {
namespace pd {

#define GET_OP_CLASSES
#include "cinnrt/dialect/pd_ops.hpp.inc"
#undef GET_OP_CLASSES

PaddleDialect::PaddleDialect(MLIRContext *context) : Dialect("pd", context, TypeID::get<PaddleDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "cinnrt/dialect/pd_ops.cpp.inc"
      >();
#undef GET_OP_LIST

  // Support unknown operations because not all Paddle operations are registered.
  // allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "cinnrt/dialect/pd_ops.cpp.inc"
#undef GET_OP_CLASSES

}  // namespace pd
}  // namespace mlir

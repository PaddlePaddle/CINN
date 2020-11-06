#include "cinnrt/dialect/cinn_base.h"

#include "cinnrt/dialect/basic_kernels.h"

namespace cinnrt::dialect {
using namespace mlir;

void CINNDialect::initialize() {
  allowUnknownTypes();
  allowUnknownOperations();

#define GET_OP_LIST
  addOperations<
#include "cinnrt/dialect/basic_kernels.cpp.inc"
      >();
#undef GET_OP_LIST
}

}  // namespace cinnrt::dialect

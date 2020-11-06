#include "cinn_base.h"

#include "basic_kernels.h"

namespace cinn::dialect {
using namespace mlir;

void CINNDialect::initialize() {
  allowUnknownTypes();
  allowUnknownOperations();

#define GET_OP_LIST
  addOperations<
#include "cinn/dialect/basic_kernels.cpp.inc"
      >();
#undef GET_OP_LIST
}

}  // namespace cinn::dialect

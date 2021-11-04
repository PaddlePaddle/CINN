#include "infrt/common/global.h"

namespace infrt {

Global::Global() {}

mlir::MLIRContext* Global::context = nullptr;

mlir::MLIRContext* Global::getMLIRContext() {
  if (nullptr == context) {
    context = new mlir::MLIRContext();
  }
  return context;
}

}  // namespace infrt

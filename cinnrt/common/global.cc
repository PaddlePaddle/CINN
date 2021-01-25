#include "cinnrt/common/global.h"

namespace cinnrt {

Global::Global() {}

mlir::MLIRContext* Global::context = nullptr;

mlir::MLIRContext* Global::getMLIRContext() {
  if (nullptr == context) {
    context = new mlir::MLIRContext();
  }
  return context;
}

}  // namespace cinnrt

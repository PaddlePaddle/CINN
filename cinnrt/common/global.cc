#include "cinnrt/common/global.h"

namespace cinnrt {

Global::Global() {}

mlir::MLIRContext* Global::context = nullptr;
TensorMap* Global::tensorMap       = nullptr;

mlir::MLIRContext* Global::getMLIRContext() {
  if (nullptr == context) {
    context = new mlir::MLIRContext();
  }
  return context;
}

TensorMap* Global::getTensorMap() {
  if (nullptr == tensorMap) {
    tensorMap = new TensorMap();
  }
  return tensorMap;
}

}  // namespace cinnrt

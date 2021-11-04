#pragma once

#include "infrt/tensor/dense_host_tensor.h"
#include "mlir/IR/MLIRContext.h"

namespace infrt {

// global variables
class Global {
 private:
  static mlir::MLIRContext *context;
  Global();

 public:
  static mlir::MLIRContext *getMLIRContext();
};  // class Global

}  // namespace infrt

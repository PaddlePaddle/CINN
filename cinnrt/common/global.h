#pragma once

#include "cinnrt/tensor/dense_host_tensor.h"
#include "mlir/IR/MLIRContext.h"

using TensorMap = std::unordered_map<std::string, cinnrt::tensor::DenseHostTensor *>;

namespace cinnrt {

// global variables
class Global {
 private:
  static mlir::MLIRContext *context;
  static TensorMap *tensorMap;
  Global();

 public:
  static mlir::MLIRContext *getMLIRContext();
  static TensorMap *getTensorMap();
};  // class Global

}  // namespace cinnrt

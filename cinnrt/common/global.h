#pragma once

#include "mlir/IR/MLIRContext.h"

namespace cinnrt {

// global variables
class Global {
private:
    static mlir::MLIRContext *context;
    Global();
public:
    static mlir::MLIRContext *getMLIRContext();
}; // class Global

} // namespace cinnrt

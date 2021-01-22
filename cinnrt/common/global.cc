#include "global.h"

namespace cinnrt {

Global::Global() {};

mlir::MLIRContext* Global::context = nullptr;

mlir::MLIRContext* Global::getMLIRContext() {
    if(nullptr == context) {
        context = new mlir::MLIRContext();
        //std::cerr << "new mlir::MLIRContext " << context << std::endl;
    }
    //std::cerr << "getMLIRContext: " << context << std::endl;
    return context;
}

} // namespace cinnrt

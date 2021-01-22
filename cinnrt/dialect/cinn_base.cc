#include "llvm/Support/raw_ostream.h"

#include "cinnrt/dialect/cinn_base.h"
#include "cinnrt/dialect/basic_kernels.h"
#include "cinnrt/dialect/dense_tensor.h"

namespace cinnrt::dialect {
using namespace mlir;

//----CINNDialect definition begin----
void CINNDialect::initialize() {
  allowUnknownTypes();
  allowUnknownOperations();

  addTypes<cinnrt::dt::TensorType>();

#define GET_OP_LIST
  addOperations<
#include "cinnrt/dialect/basic_kernels.cpp.inc"
      >();
#undef GET_OP_LIST
}

mlir::Type CINNDialect::parseType(mlir::DialectAsmParser &parser) const {
    llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return mlir::Type();
  if (keyword == "tensor") {
    llvm::StringRef target;
    llvm::StringRef layout;
    llvm::StringRef precision;

    if (parser.parseLess()) return mlir::Type();
    if (parser.parseKeyword(&target)) return mlir::Type();
    if (parser.parseComma()) return mlir::Type();
    if (parser.parseKeyword(&layout)) return mlir::Type();
    if (parser.parseComma()) return mlir::Type();
    if (parser.parseKeyword(&precision)) return mlir::Type();
    if (parser.parseGreater()) return mlir::Type();

    //llvm::outs() << target << " " << layout << " " << precision << "\n";
    return cinnrt::dt::TensorType::get(
            cinnrt::dt::getTargetType(target),
            cinnrt::dt::getLayoutType(layout),
            cinnrt::dt::getPrecisionType(precision));
  }
  parser.emitError(parser.getCurrentLocation(), "unknown cinn type: ") << keyword;
  return mlir::Type();
}

void CINNDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  if (type.isa<cinnrt::dt::TensorType>()) {
    auto tensorType = type.cast<cinnrt::dt::TensorType>();
    printer << "tensor<"
        << tensorType.getTarget()
        << ", "
        << tensorType.getLayout()
        << ", "
        << tensorType.getPrecision()
        << ">";
    return;
  }
  llvm_unreachable("unknown cinn type.");
}

//----CINNDialect definition end----

}  // namespace cinnrt::dialect

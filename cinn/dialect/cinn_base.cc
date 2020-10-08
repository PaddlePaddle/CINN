#include "cinn/dialect/cinn_base.h"

#include "cinn/dialect/basic_kernels.h"

namespace cinn::dialect {

CINN_Dialect::CINN_Dialect(mlir::MLIRContext *context) : mlir::Dialect("cinn", context) {
  allowUnknownTypes();

  allowUnknownOperations();

#define GET_OP_LIST
  addOperations<
#include "cinn/dialect/basic_kernels.cpp.inc"
      >();
}

mlir::Type CINN_Dialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (auto type = mlir::Dialect::parseType(parser)) return type;

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  ;
  mlir::emitError(loc) << "Unknown cinn type " << spec;
  return {};
}

void CINN_Dialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const { return; }

}  // namespace cinn::dialect
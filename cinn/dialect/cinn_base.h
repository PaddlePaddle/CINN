#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>

#include "cinn/dialect/cinn_base.hpp.inc"

namespace cinn::dialect {

class CINN_Dialect : public mlir::Dialect {
 public:
  explicit CINN_Dialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "cinn::dialect"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;
};

}  // namespace cinn::dialect

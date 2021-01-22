#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>

#include "cinnrt/dialect/cinn_base.hpp.inc"

namespace cinnrt::dialect {

class CINNDialect : public ::mlir::Dialect {
  explicit CINNDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<CINNDialect>()) {

    initialize();
  }

  // parse an instance of a type registered to the dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  // print an instance of a type registered to the dialect.
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static ::llvm::StringRef getDialectNamespace() { return "cinn"; }
};

}  // namespace cinnrt::dialect

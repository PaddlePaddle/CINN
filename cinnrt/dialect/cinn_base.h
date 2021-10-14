#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Types.h>

#include "cinnrt/dialect/cinn_base.hpp.inc"

namespace cinnrt::dialect {

class CINNDialect : public ::mlir::Dialect {
  explicit CINNDialect(::mlir::MLIRContext *context)
      : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<CINNDialect>()) {
    initialize();
  }

  // parse types registered to the dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  // print types registered to the dialect.
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;

  void initialize();
  friend class ::mlir::MLIRContext;

 public:
  static ::llvm::StringRef getDialectNamespace() { return "cinn"; }
};

}  // namespace cinnrt::dialect

namespace mlir {

template <typename T>
static mlir::IntegerAttr createI32Attr(mlir::OpBuilder &b, mlir::Location loc, T constant) {
  return b.getIntegerAttr(b.getI32Type(), constant);
}

}  // namespace mlir

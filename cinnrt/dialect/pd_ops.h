#pragma once

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace pd {

class PaddleDialect : public Dialect {
 public:
  explicit PaddleDialect(MLIRContext* context);

  static StringRef getDialectNamespace() { return "PD"; }

  Type parseType(DialectAsmParser& parser) const override { return Dialect::parseType(parser); }
  void printType(Type type, DialectAsmPrinter& printer) const override { Dialect::printType(type, printer); }
};

}  // namespace pd
}  // namespace mlir

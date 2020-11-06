#include "cinnrt/dialect/tensor_shape.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LogicalResult.h>

namespace cinn::ts {
using namespace mlir;

void TensorShapeDialect::initialize() {
  allowUnknownTypes();
  addTypes<ShapeType, PartialShapeType>();
  addOperations<
#define GET_OP_LIST
#include "cinn/dialect/tensor_shape.cpp.inc"
      >();
}

Type TensorShapeDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();
  if (keyword == "shape") return ShapeType::get(getContext());
  if (keyword == "partial_shape") return PartialShapeType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown shape type: ") << keyword;
  return Type();
}

void TensorShapeDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &os) const {
  if (type.isa<ShapeType>()) {
    os << "shape";
    return;
  }

  if (type.isa<PartialShapeType>()) {
    os << "partial_shape";
    return;
  }
  llvm_unreachable("unexpected 'shape' type kind");
}

#define GET_OP_CLASSES
#include "cinnrt/dialect/tensor_shape.cpp.inc"

}  // namespace cinn::ts

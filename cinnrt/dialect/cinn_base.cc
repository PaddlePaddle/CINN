#include "infrt/dialect/cinn_base.h"

#include "infrt/dialect/basic_kernels.h"
#include "infrt/dialect/dense_tensor.h"
#include "infrt/dialect/test_kernels.h"

namespace infrt::dialect {

// ----CINNDialect definition begin----
void CINNDialect::initialize() {
  allowUnknownTypes();
  allowUnknownOperations();

  addTypes<infrt::dt::StringType>();
  addTypes<infrt::dt::TensorType>();
  addTypes<infrt::dt::TensorMapType>();

  addOperations<
#define GET_OP_LIST
#include "infrt/dialect/basic_kernels.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "infrt/dialect/test_kernels.cpp.inc"
      >();
}

mlir::Type CINNDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return mlir::Type();
  // parse TensorType, for example: !cinn.tensor<X86, CUDA, F32>
  if (keyword == "tensor") {
    llvm::StringRef target;
    llvm::StringRef layout;
    llvm::StringRef precision;

    // parse "<"
    if (parser.parseLess()) return mlir::Type();
    // parse target
    if (parser.parseKeyword(&target)) return mlir::Type();
    auto targetType = infrt::dt::GetTargetType(target);
    if (!targetType) {
      parser.emitError(parser.getCurrentLocation(), "unknown target type: ") << target;
      return mlir::Type();
    }

    // parse ","
    if (parser.parseComma()) return mlir::Type();
    // parse layout
    if (parser.parseKeyword(&layout)) return mlir::Type();
    auto layoutType = infrt::dt::GetLayoutType(layout);
    if (!layoutType) {
      parser.emitError(parser.getCurrentLocation(), "unknown layout type: ") << layout;
      return mlir::Type();
    }

    // parse ","
    if (parser.parseComma()) return mlir::Type();
    // parse precision
    if (parser.parseKeyword(&precision)) return mlir::Type();
    auto precisionType = infrt::dt::GetPrecisionType(precision);
    if (!precisionType) {
      parser.emitError(parser.getCurrentLocation(), "unknown precision type: ") << precision;
      return mlir::Type();
    }

    // parse ">"
    if (parser.parseGreater()) return mlir::Type();

    return infrt::dt::TensorType::get(*targetType, *layoutType, *precisionType);
  }
  // parse TensorMapType, for example: !cinn.tensor_map
  if (keyword == "tensor_map") {
    return infrt::dt::TensorMapType::get();
  }
  // parse StringType, for example: !cinn.string
  if (keyword == "string") {
    return infrt::dt::StringType::get();
  }

  parser.emitError(parser.getCurrentLocation(), "unknown cinn type: ") << keyword;
  return mlir::Type();
}

void CINNDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  // print TensorType, for example: !cinn.tensor<X86, CUDA, F32>
  if (type.isa<infrt::dt::TensorType>()) {
    auto tensorType = type.cast<infrt::dt::TensorType>();
    printer << "tensor<" << tensorType.target() << ", " << tensorType.layout() << ", " << tensorType.precision() << ">";
    return;
  }
  // print TensorMapType, for example: !cinn.tensor_map
  if (type.isa<infrt::dt::TensorMapType>()) {
    printer << "tensor_map";
    return;
  }
  // print StringType, for example: !cinn.string
  if (type.isa<infrt::dt::StringType>()) {
    printer << "string";
    return;
  }
  llvm_unreachable("unknown cinn type.");
}

// ----CINNDialect definition end----

}  // namespace infrt::dialect

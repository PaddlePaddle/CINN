#include "cinnrt/dialect/dense_tensor.h"

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

#include "cinnrt/dialect/tensor_shape.h"

namespace cinn::dt {
using namespace mlir;

void DTDialect::initialize() {
  allowUnknownTypes();
  addTypes<TensorType>();
  addOperations<
#define GET_OP_LIST
#include "cinn/dialect/dense_tensor.cpp.inc"
      >();
}

static Type getTensorType(mlir::MLIRContext* context) {
  auto t_dialect = Identifier::get("t", context);
  return OpaqueType::get(t_dialect, "tensor", context);
}

static ParseResult parseCreateUninitTensorOp(OpAsmParser& parser, OperationState& result) {
  Type attr_type   = IntegerType::get(64, result.getContext());
  auto tensor_type = getTensorType(result.getContext());

  Attribute value_attr;
  return failure(parser.parseAttribute(value_attr, attr_type, "shape", result.attributes) ||
                 parser.addTypeToList(tensor_type, result.types));
}

template <typename CreateUninitTensorOp>
static void printCreateUninitTensorOp(OpAsmPrinter& p, CreateUninitTensorOp op) {
  p << CreateUninitTensorOp::getOperationName() << " " << op.getAttr("shape");
}

static ParseResult parseFillTensorWithConstantOp(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::OperandType, 1> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/1)) return failure();

  auto tensor_type = getTensorType(result.getContext());

  Attribute value_attr;
  return failure(parser.resolveOperand(operands[0], tensor_type, result.operands) ||
                 parser.parseAttribute(value_attr, "value", result.attributes));
}

template <typename FillTensorOp>
static void printFillTensorWithConstantOp(OpAsmPrinter& p, FillTensorOp op) {
  p << FillTensorOp::getOperationName() << " ";
  p.printOperand(op.getOperand());
  p << " " << op.getAttr("value");
}

static ParseResult parseSetTensorOp(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::OperandType, 1> operands;
  if (parser.parseOperandList(operands, 1)) return failure();

  auto tensor_type = getTensorType(result.getContext());

  Attribute value_attr;
  return failure(parser.resolveOperand(operands[0], tensor_type, result.operands) ||
                 parser.parseAttribute(value_attr, "values", result.attributes));
}

template <typename SetTensorOp>
static void printSetTensorOp(OpAsmPrinter& p, SetTensorOp op) {
  p << SetTensorOp::getOperationName() << " ";
  p.printOperand(op.getOperand());
  p << " " << op.getAttr("values");
}

#define GET_OP_CLASSES
#include "cinn/dialect/dense_tensor.cpp.inc"

}  // namespace cinn::dt

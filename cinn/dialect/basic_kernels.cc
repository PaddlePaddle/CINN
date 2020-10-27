#include "cinn/dialect/basic_kernels.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LogicalResult.h>

namespace cinn::dialect {
using namespace mlir;

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {
  SymbolRefAttr callee_attr;
  FunctionType callee_type;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  auto callee_loc = parser.getNameLoc();
  if (parser.parseAttribute(callee_attr, "callee", result.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(callee_type) ||
      parser.addTypesToList(callee_type.getResults(), result.types) ||
      parser.resolveOperands(operands, callee_type.getInputs(), callee_loc, result.operands))
    return failure();
  return success();
}

static ParseResult parseConstantOp(Type attrType, OpAsmParser &parser, OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, attrType, "value", result.attributes) ||
      parser.addTypeToList(attrType, result.types))
    return failure();
  return success();
}

static ParseResult parseConstantF32Op(OpAsmParser &parser, OperationState &result) {
  return parseConstantOp(FloatType::getF32(result.getContext()), parser, result);
}
static ParseResult parseConstantF64Op(OpAsmParser &parser, OperationState &result) {
  return parseConstantOp(FloatType::getF64(result.getContext()), parser, result);
}
static ParseResult parseConstantI32Op(OpAsmParser &parser, OperationState &result) {
  return parseConstantOp(IntegerType::get(32, result.getContext()), parser, result);
}
static ParseResult parseConstantI64Op(OpAsmParser &parser, OperationState &result) {
  return parseConstantOp(IntegerType::get(64, result.getContext()), parser, result);
}

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) || (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

static void print(OpAsmPrinter &p, CallOp op) {
  p << "cinn.call " << op.getAttr("callee") << "(";
  p.printOperands(op.getOperands());
  p << ")";
  p.printOptionalAttrDict(op.getAttrs(), {"callee"});
  p << " : ";
}

static void printConstant(OpAsmPrinter &p, mlir::Operation *op) {
  p << op->getName() << " ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  if (op->getAttrs().size() > 1) p << ' ';
  Attribute attr = op->getAttr("value");
  if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
    bool is_signed = int_attr.getType().isIndex() || int_attr.getType().getIntOrFloatBitWidth() != 1;
    int_attr.getValue().print(p.getStream(), is_signed);
  } else if (auto float_attr = attr.dyn_cast<FloatAttr>()) {
    p << float_attr.getValue().convertToFloat();
  } else {
    op->emitOpError("unknown attribute type");
  }
}

static void print(OpAsmPrinter &p, ConstantF32Op op) { printConstant(p, op); }
static void print(OpAsmPrinter &p, ConstantF64Op op) { printConstant(p, op); }
static void print(OpAsmPrinter &p, ConstantI32Op op) { printConstant(p, op); }
static void print(OpAsmPrinter &p, ConstantI64Op op) { printConstant(p, op); }

static void print(OpAsmPrinter &p, ReturnOp op) {
  p << "cinn.return";
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.getOperands());
    p << " : ";
    llvm::interleaveComma(op.getOperands(), p);
  }
}

static LogicalResult verify(CallOp op) { return success(); }

static LogicalResult verify(ConstantF32Op op) { return success(); }
static LogicalResult verify(ConstantI32Op op) { return success(); }
static LogicalResult verify(ConstantF64Op op) { return success(); }
static LogicalResult verify(ConstantI64Op op) { return success(); }

static LogicalResult verify(ReturnOp op) {
  auto function = dyn_cast<FuncOp>(op.getParentOp());

  if (!function) return success();

  auto results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ") << op.getNumOperands() << " operands, but enclosing function returns "
                                  << results.size();

  return success();
}

#define GET_OP_CLASSES
#include "cinn/dialect/basic_kernels.cpp.inc"

}  // namespace cinn::dialect

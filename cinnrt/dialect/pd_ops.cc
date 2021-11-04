#include "infrt/dialect/pd_ops.h"

#include "infrt/dialect/cinn_base.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pd {

#define GET_OP_CLASSES
#include "infrt/dialect/pd_ops.hpp.inc"
#undef GET_OP_CLASSES

PaddleDialect::PaddleDialect(MLIRContext *context) : Dialect("pd", context, TypeID::get<PaddleDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "infrt/dialect/pd_ops.cpp.inc"
      >();
#undef GET_OP_LIST
}

mlir::Operation *PaddleDialect::materializeConstant(mlir::OpBuilder &builder,
                                                    mlir::Attribute value,
                                                    mlir::Type type,
                                                    mlir::Location loc) {
  return builder.create<ConstantOp>(loc, value);
}

#define GET_OP_CLASSES
#include "infrt/dialect/pd_ops.cpp.inc"
#undef GET_OP_CLASSES

#include "infrt/dialect/rewrite.hpp.inc"

void ConstantOp::build(OpBuilder &builder, OperationState &state, Attribute value) {
  if (auto elem_attr = value.dyn_cast<ElementsAttr>()) {
    return ConstantOp::build(builder, state, elem_attr);
  } else if (value.isa<BoolAttr, FloatAttr, IntegerAttr>()) {
    ShapedType type = RankedTensorType::get(/*shape=*/{}, value.getType());
    state.addAttribute("value", DenseElementsAttr::get(type, value));
    state.addTypes(type);
    return;
  }
  llvm_unreachable("unsupported attribute type for building pd.constant");
}

LogicalResult ConstantOp::inferReturnTypes(MLIRContext *context,
                                           Optional<Location> location,
                                           ValueRange operands,
                                           DictionaryAttr attributes,
                                           RegionRange regions,
                                           SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(attributes.get("value").getType());
  return success();
}
::mlir::OpFoldResult ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands) { return value(); }

LogicalResult ElementwiseAdd::inferReturnTypes(MLIRContext *context,
                                               Optional<Location> location,
                                               ValueRange operands,
                                               DictionaryAttr attributes,
                                               RegionRange regions,
                                               SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}
void ElementwiseAdd::getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results,
                                                 ::mlir::MLIRContext *context) {
  results.insert<FuseMulAdd>(context);
}

::mlir::OpFoldResult ElementwiseAdd::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  if (getElementTypeOrSelf(getType()).isa<FloatType>()) {
    if (!operands[0] || !operands[1]) return {};
    DenseElementsAttr lhs = operands[0].dyn_cast<DenseElementsAttr>();
    DenseElementsAttr rhs = operands[1].dyn_cast<DenseElementsAttr>();
    if (!lhs || !rhs) return {};
    ShapedType type = getType().template cast<ShapedType>();
    if (!type.hasStaticShape()) return {};
    Type etype = type.getElementType();
    if (!etype.isa<FloatType>()) return {};
    SmallVector<APFloat, 6> values;
    values.reserve(lhs.getNumElements());
    for (const auto zip : llvm::zip(lhs.getValues<APFloat>(), rhs.getValues<APFloat>())) {
      values.push_back(std::plus<APFloat>()(std::get<0>(zip), std::get<1>(zip)));
    }
    return DenseElementsAttr::get(type, values);
  }
  return {};
}

LogicalResult ElementwiseDiv::inferReturnTypes(MLIRContext *context,
                                               Optional<Location> location,
                                               ValueRange operands,
                                               DictionaryAttr attributes,
                                               RegionRange regions,
                                               SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

LogicalResult ElementwiseMul::inferReturnTypes(MLIRContext *context,
                                               Optional<Location> location,
                                               ValueRange operands,
                                               DictionaryAttr attributes,
                                               RegionRange regions,
                                               SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

LogicalResult ElementwiseSub::inferReturnTypes(MLIRContext *context,
                                               Optional<Location> location,
                                               ValueRange operands,
                                               DictionaryAttr attributes,
                                               RegionRange regions,
                                               SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

LogicalResult MulOp::inferReturnTypes(MLIRContext *context,
                                      Optional<Location> location,
                                      ValueRange operands,
                                      DictionaryAttr attributes,
                                      RegionRange regions,
                                      SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

void ReluOp::getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
  results.insert<FuseFCRelu>(context);
}

void FusedRepeatedFCRelu::getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results,
                                                      ::mlir::MLIRContext *context) {
  results.insert<FuseRepeatedFCRelu2>(context);
}

void BatchNormOp::getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
  results.insert<FuseBatchNormWithConvPattern>(context);
}

}  // namespace pd
}  // namespace mlir

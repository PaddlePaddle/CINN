#include "cinnrt/dialect/pd_ops.h"

#include "cinnrt/dialect/cinn_base.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pd {

#define GET_OP_CLASSES
#include "cinnrt/dialect/pd_ops.hpp.inc"
#undef GET_OP_CLASSES

PaddleDialect::PaddleDialect(MLIRContext *context) : Dialect("pd", context, TypeID::get<PaddleDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "cinnrt/dialect/pd_ops.cpp.inc"
      >();
#undef GET_OP_LIST

  // Support unknown operations because not all Paddle operations are registered.
  // allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "cinnrt/dialect/pd_ops.cpp.inc"
#undef GET_OP_CLASSES

#include "cinnrt/dialect/rewrite.hpp.inc"

void ElementwiseAdd::getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results,
                                                 ::mlir::MLIRContext *context) {
  results.insert<FuseMulAdd>(context);
}

void ReluOp::getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
  results.insert<FuseFCRelu>(context);
}

void FusedRepeatedFCRelu::getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results,
                                                      ::mlir::MLIRContext *context) {
  results.insert<FuseRepeatedFCRelu2>(context);
}

}  // namespace pd
}  // namespace mlir

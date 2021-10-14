#include "cinnrt/dialect/pd_ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pd {

#define GET_OP_CLASSES
#include "cinnrt/dialect/pd_ops.hpp.inc"
#undef GET_OP_CLASSES

}  // namespace pd
}  // namespace mlir

namespace {
using namespace mlir;
::mlir::IntegerAttr createI32Attr(::mlir::Builder &builder, int32_t constant) {
  return builder.getI32IntegerAttr(constant);
}
/// Include the patterns defined in the Declarative Rewrite framework.
#include "cinnrt/dialect/fc_fuse_pattern.inc"
}  // end anonymous namespace
namespace mlir {

namespace pd {

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void ElementwiseAdd::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FcFusePattern>(context);
}

}  // namespace pd
}  // namespace mlir
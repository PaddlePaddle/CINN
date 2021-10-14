#ifndef FC_FUSE_PATTERN
#define FC_FUSE_PATTERN

include "cinnrt/dialect/pd_ops.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// ElementwiseAdd(Matmul(x)) = Fc(x)

class createI32Attr<string value> : NativeCodeCall<"createI32Attr($_builder, " # value # ")">;

def FcFusePattern : Pat<(PD_ElementwiseAdd (PD_MatmulOp $arg, $arg1, $_, $_, $_), $arg2, $_),(PD_FcOp $arg, $arg1, $arg2, (createI32Attr<"1">))>;

#endif // FC_FUSE_PATTERN

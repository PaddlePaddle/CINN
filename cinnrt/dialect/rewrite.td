// Operation definitions for basic kernels.

#ifdef CINNRT_REWRITE
#else
#define CINNRT_REWRITE

#ifndef CINN_BASE
include "cinnrt/dialect/cinn_base.td"
#endif
include "mlir/Interfaces/SideEffectInterfaces.td"
#ifndef PD_OPS
include "cinnrt/dialect/pd_ops.td"
#endif

def FuseMulAdd : Pat<(PD_ElementwiseAdd (PD_MatmulOp:$out $x, $y, $transpose_x, $transpose_y, $alpha), $bias, $axis),
                     (PD_FusedFC $x, $y, $bias,
                     (CINN_createI32Attr<"1">))>;

#endif // CINNRT

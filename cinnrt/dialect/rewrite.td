#ifndef CINNRT_REWRITE
#define CINNRT_REWRITE

include "cinnrt/dialect/cinn_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "cinnrt/dialect/pd_ops.td"

//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'Matmul o ElementwiseAdd' into 'PD_FusedFC'.
//
// We have:
//   (Matmul)      z = x * y
//   (Add)         out = z + bias 
//
// which corresponds to the following computation:
//   (FusedFC)  out = x * y + bias
// 
// Todo:
//  1. Make the constrait more completely.
//  2. Consider the case of : out = bias + z
//===----------------------------------------------------------------------===//
def FuseMulAdd : Pat<(PD_ElementwiseAdd (PD_MatmulOp $x, $y, $transpose_x, $transpose_y, $alpha), $bias, $axis),
                     (PD_FusedFC $x, $y, $bias, (CINN_createI32Attr<"1">)),
                     [(IsBoolAttrEq<"false"> $transpose_x),(IsBoolAttrEq<"false"> $transpose_y)]>;


//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'FusedFC o Relu' into 'FusedRepeatedFCRelu'.
//
// We have:
//   (FusedFC)      z = fc(x, y, bias)
//   (Relu)         out = relu(z)
//
// which corresponds to the following computation:
//   (FusedRepeatedFCRelu)  out = RepeatedFCRelu(x, [y], [bias])
// 
//===----------------------------------------------------------------------===//
def FuseFCRelu : Pat<(PD_ReluOp (PD_FusedFC $x, $y, $bias, $_)),
                     (PD_FusedRepeatedFCRelu $x, (CINN_cvtValueToValueRange $y), (CINN_cvtValueToValueRange $bias))>;

//===----------------------------------------------------------------------===//
// This is to fold 'FusedRepeatedFCRelu' op.
//
// We have:
//   (FusedRepeatedFCRelu)      z = RepeatedFCRelu(x, [y, ...], [bias, ...])
//   (FusedRepeatedFCRelu)      out = RepeatedFCRelu(z, [y1, ...], [bias1, ...])
//
// which corresponds to the following computation:
//   (FusedRepeatedFCRelu)  out = RepeatedFCRelu(x, [y, ..., y1, ...], [bias, ..., bias1, ....])
// 
//===----------------------------------------------------------------------===//
def FuseRepeatedFCRelu2 : Pat<(PD_FusedRepeatedFCRelu (PD_FusedRepeatedFCRelu $x, $y, $bias), $y_2, $bias_2),
                     (PD_FusedRepeatedFCRelu $x, (CINN_concatTwoValueRange $y, $y_2), (CINN_concatTwoValueRange $bias, $bias_2))>;


//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'BatchNorm o Conv' into 'Conv'
// by deriving new 'w' and 'b' for 'Conv':
//
// We have:
//   (Conv)      z = w * x + b 
//   (BatchNorm) y = scale * (z - mean) / sqrt(var + eps) + bias
//
// which corresponds to the following computation:
//   y = w_ * x + b_
// where
//   w_ = scale * w / sqrt(var + eps)
//   b_ = B + scale * (b - mean) / sqrt(var + eps)
//
// Hence, we rewrite: 
//   onnx.BatchNormalizationInferenceMode(
//       onnx.Conv(x, w, b),
//       scale, B, mean, var
//   ) {eps = ...}
//
// as:
//    onnx.Conv(x, w_, b_)
//    
//    where
//      w_ = scale * w / sqrt(var + eps)
//      b_ = B + scale * (b - mean) / sqrt(var + eps)
//
//===----------------------------------------------------------------------===//
//def FuseBatchNormWithConvPattern: Pat<
//  (BatchNorm:$res
//    (ConvOp $x, $w, $b,
//                $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
//    $scale, $B, $mean, $var, $epsilon, $momentum),
//  (ConvOp
//     $x,
//     // w_
//     (MulOp
//        $w,
//        (UnsqueezeV11Op
//           (DivOp:$coefficientW
//              $scale,
//              (SqrtOp
//                (AddOp
//                    $var,
//                    (ConstantOpFromDenseAttr
//                       (createDenseElementsAttrFromFloatAttr $res, $epsilon))))),
//           (createArrayAttrOfOneToRankOf $w))),
//    // b_
//     (AddOp
//        $B,
//        (MulOp
//           $coefficientW,
//           (subtractOrNeg $res, $b, $mean))),
//
//     $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
//     [], (addBenefit 1)
//>;

#endif // CINNRT_REWRITE

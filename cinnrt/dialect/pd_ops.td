#ifndef PD_OPS
#define PD_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/OpBase.td"
include "cinnrt/dialect/pd_op_base.td"

def PD_FeedOp : PD_Op<"Feed", [NoSideEffect]> {
  let summary = "Feed Op";

  let description = [{
    Feed a tensor into the model.
  }];

  let arguments = (ins);
  let results = (outs PD_Tensor:$out);

  let assemblyFormat = [{
      `(` `)` attr-dict `:` type($out)
  }];
}

def PD_AbsOp : PD_Op<"Abs", [NoSideEffect]> {
  let summary = "Computes the absolute value of a tensor";

  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x);
  let results = (outs PD_Tensor:$y);
}

def PD_ReluOp : PD_Op<"Relu", [NoSideEffect]> {
  let summary = "Computes the Relu of a tensor";

  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x);
  let results = (outs PD_Tensor:$y);
}

def PD_Relu6Op : PD_Op<"Relu6", [NoSideEffect]> {
  let summary = "Computes the Relu6 of a tensor";

  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x);
  let results = (outs PD_Tensor:$y);
}

def PD_ElementwiseAdd : PD_Op<"ElementwiseAdd", [NoSideEffect]> {
  let summary = "ElementwiseAdd Op";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y, I32Attr:$axis);
  let results = (outs PD_Tensor:$out);

  let hasCanonicalizer = 1;
}

def PD_ElementwiseMul : PD_Op<"ElementwiseMul", [NoSideEffect]> {
  let summary = "ElementwiseMul Op";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y, I32Attr:$axis);
  let results = (outs PD_Tensor:$out);
}

def PD_MatmulOp : PD_Op<"Matmul", [NoSideEffect]> {
  let summary = "Computes the matrix mulplication result of two tensors";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y,
                  DefaultValuedAttr<BoolAttr, "false">:$transpose_x,
                  DefaultValuedAttr<BoolAttr, "false">:$transpose_y,
                  DefaultValuedAttr<F32Attr, "1.0">:$alpha);
  let results = (outs PD_Tensor:$out);

  //let hasCanonicalizer = 1;
}

def PD_FcOp : PD_Op<"Fc", [NoSideEffect]> {
  let summary = "Computes the fully connect result of three tensors";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y, PD_Tensor:$bias,
                   DefaultValuedAttr<I32Attr, "1">:$in_num_col_dims);
  let results = (outs PD_Tensor:$out);

  //let hasCanonicalizer = 1;
}
#endif  // PD_OPS

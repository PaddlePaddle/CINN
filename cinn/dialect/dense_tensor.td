include "cinn/dialect/cinn_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DenseTensor_Dialect : Dialect {
  let name = "cinn_dt";

  let description = [{
      The DenseTensor dialect.
  }];

  let cppNamespace = "cinn::dialect::dt";
}

class DT_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<DT_Dialect, mnemonic, traits>;

class CreateUninitTensorOp<string dtype, int rank>
      : DT_Op<"create_uninit_tensor." # dtype # "." # rank, [NoSideEffect]> {
  let summary = "cinn_dt.create_uninit_tensor operation";

  let description = [{
      An operation that creates an uninitialized tensor.
  }];

  let arguments = (ins);  // no input arguments but a attribute
  let results = (outs TensorType);

  let printer = [{ return cinn::dialect::dt::printCreateUninitTensorOp(p, *this); }];
  let parser = " return cinn::dialect::dt::parseCreateUninitTensorOp(parser, result);";
}

class MakeTensorOp<string dtype> : DT_Op<"make_tensor." # dtype> {
  let summary = "cinn_dt.make_tensor operation";
  let description = [{}];
}

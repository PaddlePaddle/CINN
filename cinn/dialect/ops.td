include "mlir/IR/OpBase.td"

def CINN_Dialect : Dialect {
  let name = "cinn";
  let cppNamespace = "::cinn::dialect";
}

class CINN_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<CINN_Dialect, mnemonic, traits>;

def StringType : OpaqueType<"cinn", "string", "!cinn.string type">;

def TensorType : OpaqueType<"t", "tensor", "tensor">;

#ifdef CINN_BASE
#else
#define CINN_BASE

include "mlir/IR/OpBase.td"

def CINN_Dialect : Dialect {
  let name = "cinn";

  let description = [{
    The CINN host dialect.
  }];

  let cppNamespace = "::cinn::dialect";
}



// Type definitions
def StringType : OpaqueType<"cinn", "string", "!cinn.string type">;

def TensorType : OpaqueType<"t", "tensor", "tensor">;


#endif  // CINN_BASE
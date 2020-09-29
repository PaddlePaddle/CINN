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

#endif  // CINN_BASE
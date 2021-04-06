#ifdef CINN_BASE
#else
#define CINN_BASE

include "mlir/IR/OpBase.td"

def CINN_Dialect : Dialect {
  let name = "cinn";

  let description = [{
    The CINN host dialect.
  }];

  let cppNamespace = "::cinnrt::dialect";
}

// Type definitions
def StringType :
    Type<CPred<"$_self.isa<::cinnrt::dt::StringType>()">, "!tfrt.string type">,
    BuildableType<"$_builder.getType<::cinnrt::dt::StringType>()">;

def TensorType :
    Type<CPred<"$_self.isa<::cinnrt::dt::TensorType>()">, "!cinn.tensor type">;

def TensorMapType :
    Type<CPred<"$_self.isa<::cinnrt::dt::TensorMapType>()">, "!cinn.tensor_map type">,
    BuildableType<"$_builder.getType<::cinnrt::dt::TensorMapType>()">;

def BufferType : OpaqueType<"b", "buffer", "buffer">;

#endif  // CINN_BASE

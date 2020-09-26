include "mlir/IR/OpBase.td"

def CINN_Dialect : Dialect {
  let name = "cinn";
  let cppNamespace = "cinn::hlir::dialect";
}

class CINN_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<CINN_Dialect, mnemonic, traits>;

def ONNXEntryPointOp: CINN_Op<"EntryPoint"> {
  let summary = "Indicate ONNX entry point";
  let description = [{
  }];

  let builders = [OpBuilder<[{OpBuilder &builder, OperationState &state,
                             FuncOp function, int numInputs, int numOutputs}]>];

  let extraClassDeclaration = [{
    static ONNXEntryPointOp create(Location location, FuncOp& func,
                                   int numInputs, int numOutputs);

    static StringRef getEntryPointFuncAttrName() { return "func"; }
    static StringRef getNumInputsAttrName() { return "numInputs"; }
    static StringRef getNumOutputsAttrName() { return "numOutputs"; }
  }];
}
#pragma once

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/OperationSupport.h>
#include <memory>
#include <string>
#include <unordered_map>

#include "cinnrt/host_context/core_runtime.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/mlir_function_executable.h"
#include "cinnrt/host_context/mlir_to_runtime_translate.h"
#include "cinnrt/host_context/op_executable.h"

namespace cinnrt {
namespace host_context {

/**
 * This get a MLIR program as input, it compiles it into runtime program, and one can retrieve the function and execute
 * it by passing the input arguments.
 */
class MlirProgramExecutor : public MlirToRuntimeTranslator {
 public:
  CoreRuntimeBuilder runtime_builder;
  mlir::ModuleOp module;
  function_defs_t function_defs;

  MlirProgramExecutor(mlir::ModuleOp module, KernelRegistry* registry)
      : runtime_builder(registry), MlirToRuntimeTranslator(module, &runtime_builder), module(module) {}

  // Build functions and generate executables.
  void BuildFunctions() { EmitFunctions(); }

  void EmitFunction(mlir::FuncOp op) override {
    LOG(INFO) << "Emit function: " << op.getName().str();
    function_defs[op.getName().str()] = op;

    func_executables_.emplace(op.getName().str(),
                              new MlirFunctionExecutable(op, runtime_builder.kernel_registry(), function_defs));
  }

  MlirFunctionExecutable* LookupFunc(const std::string& name) {
    auto it = func_executables_.find(name);
    if (it != func_executables_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<MlirFunctionExecutable>> func_executables_;
};

}  // namespace host_context
}  // namespace cinnrt

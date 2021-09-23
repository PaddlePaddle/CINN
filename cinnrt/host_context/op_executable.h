#pragma once
#include <llvm/ADT/ArrayRef.h>

#include <memory>
#include <string>
#include "absl/strings/string_view.h"
#include "absl/container/flat_hash_map.h"

#include "mlir/IR/Function.h"
#include "mlir/IR/Region.h"

namespace mlir {
class FuncOp;
}  // namespace mlir

namespace cinnrt::host_context {

class SymbolTable;
class KernelRegistry;
class KernelFrame;
class Value;
class CoreRuntimeBuilder;
class MlirFunctionExecutable;

/**
 * OpExecutable is a runtime executable instance for an operation. It captures all the information(Tensors, attributes
 * and so on) needed for execution.
 * With the SymbolTable and op definition, it create and hold a KernelFrame once and execute any times.
 */
class OpExecutable {
 public:
  KernelFrame& frame();
  const KernelFrame& frame() const;

  void Execute();

  absl::string_view name() const;

  ~OpExecutable();

 protected:
  class Impl;
  explicit OpExecutable(Impl* impl);

  std::unique_ptr<Impl> impl_;
};

/**
 * Builder to help contruct an OpExecutable.
 */
class OpExecutableBuilder : public OpExecutable {
 public:
  using function_defs_t = absl::flat_hash_map<std::string, mlir::FuncOp>;

  OpExecutableBuilder(absl::string_view op_name, SymbolTable* symbol_table, KernelRegistry* kernel_registry = nullptr);
  OpExecutableBuilder(OpExecutableBuilder&& other);

  void AppendArgument(absl::string_view name);
  void AppendArgument(Value* value);

  void SetResults(llvm::ArrayRef<std::string> result_names);
  void SetResults(llvm::ArrayRef<Value*> results);

  void AppendAttribute(Value* value);

  MlirFunctionExecutable* CreateFunctionExecutable(mlir::FuncOp op, function_defs_t* function_defs);

  MlirFunctionExecutable* CreateFunctionExecutable(mlir::Region* region,
                                                   mlir::FunctionType func_type,
                                                   function_defs_t* function_defs);

  //! Get the CoreRuntime instance for function call(used in `cinn.call` op).
  CoreRuntimeBuilder* GetCallRuntimeBuilder();
};

}  // namespace cinnrt::host_context

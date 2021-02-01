#pragma once
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Function.h>

#include <memory>
#include <string>
#include <string_view>

#include "cinnrt/host_context/mlir_to_runtime_translate.h"

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
 *
 * An OpExecutable is an item of a CoreRuntime, but it can holds a CoreRuntime instance for function call (e.g. a
 * `cinn.call` op).
 */
class OpExecutable {
 public:
  KernelFrame& frame();
  const KernelFrame& frame() const;

  void Execute();

  std::string_view name() const;

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
  OpExecutableBuilder(std::string_view op_name, SymbolTable* symbol_table, KernelRegistry* kernel_registry = nullptr);
  OpExecutableBuilder(OpExecutableBuilder&& other);

  void AppendArgument(std::string_view name);
  void AppendArgument(Value* value);

  void SetResults(llvm::ArrayRef<std::string> result_names);
  void SetResults(llvm::ArrayRef<Value*> results);

  void AppendAttribute(Value* value);

  MlirFunctionExecutable* CreateFunctionExecutable(mlir::FuncOp op,
                                                   MlirToRuntimeTranslator::function_defs_t* function_defs);

  //! Get the CoreRuntime instance for function call(used in `cinn.call` op).
  CoreRuntimeBuilder* GetCallRuntimeBuilder();
};

}  // namespace cinnrt::host_context

#pragma once
#include <llvm/ADT/ArrayRef.h>

#include <memory>
#include <string>
#include <string_view>

namespace cinn::host_context {

class SymbolTable;
class KernelRegistry;
class KernelFrame;
class Value;

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

  ~OpExecutable();

 protected:
  class Impl;
  OpExecutable(Impl* impl);

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
};

}  // namespace cinn::host_context

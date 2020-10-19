#pragma once
#include <llvm/ADT/ArrayRef.h>

#include <memory>
#include <string>
#include <string_view>

namespace cinn::host_context {

class SymbolTable;
class KernelRegistry;
class KernelFrame;

/**
 * OpExecutable is a runtime executable instance for an operation. It captures all the information(Tensors, attributes
 * and so on) needed for execution.
 * With the SymbolTable and op definition, it create and hold a KernelFrame once and execute any times.
 */
class OpExecutable {
 public:
  OpExecutable(std::string_view op_name, SymbolTable* symbol_table, KernelRegistry* kernel_registry = nullptr);

  void AppendArgument(std::string_view name);

  void SetResults(llvm::ArrayRef<std::string> result_names);

  void SetResultNum(int num);

  KernelFrame& frame();
  const KernelFrame& frame() const;

  void Execute();

  ~OpExecutable();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cinn::host_context

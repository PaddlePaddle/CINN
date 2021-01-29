#pragma once
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include "cinnrt/host_context/value.h"

#include <memory>
#include <string>

namespace cinnrt::host_context {

class KernelRegistry;
class OpExecutable;
class OpExecutableBuilder;
class SymbolTable;

/**
 * CoreRuntime encapsulate the execution for a function.
 * Each function call will bind to a CoreRuntime instance, push the argument Values in to the argument-list, and get the
 * result Values from the return-list.
 */
class CoreRuntime {
 public:
  //! Execute a program.
  void Execute();

  size_t num_ops() const;

  //! Feed the input arguments, each item is a pair of arg-name and arg-value.
  void FeedInArgs(llvm::ArrayRef<std::pair<std::string, ValueRef>> args);

  //! Get the results of the execution.
  llvm::SmallVector<ValueRef, 4> GetResults(llvm::ArrayRef<std::string_view> arg_names);

  ~CoreRuntime();

 protected:
  //! Get the symbol table.
  SymbolTable* symbol_table();

  class Impl;
  explicit CoreRuntime(Impl* impl);
  std::unique_ptr<Impl> impl_;
};

/**
 * The builder for CoreRuntime, help to construct a function.
 */
class CoreRuntimeBuilder : public CoreRuntime {
 public:
  explicit CoreRuntimeBuilder(KernelRegistry* kernel_registry);

  using CoreRuntime::symbol_table;

  llvm::ArrayRef<std::string_view> attr_names() const;

  OpExecutableBuilder* NewOpExecutable(std::string_view op_name, const std::string& fn_name);
};

}  // namespace cinnrt::host_context

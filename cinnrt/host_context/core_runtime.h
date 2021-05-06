#pragma once
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <string>
#include <utility>
#include "cinnrt/host_context/value.h"

namespace cinnrt::host_context {

class KernelRegistry;
class OpExecutable;
class OpExecutableBuilder;
class SymbolTable;

/**
 * CoreRuntime encapsulate the execution for a sequence of ops.
 * Each function call will bind to a CoreRuntime instance, push the argument Values in to the argument-list, and get the
 * result Values from the return-list.
 */
class CoreRuntime : public std::enable_shared_from_this<CoreRuntime> {
 public:
  //! Execute a program.
  void Execute();

  //! Return the number of ops.
  size_t num_ops() const;

  //! Get the results of the execution.
  llvm::SmallVector<ValueRef, 4>  //
  GetResults(llvm::ArrayRef<std::string_view> arg_names);

  std::shared_ptr<CoreRuntime> getptr() { return std::shared_ptr<CoreRuntime>(this); }

  KernelRegistry* kernel_registry() const;

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

  void SetKernelRegistry(KernelRegistry* x);

  //! Feed the input arguments, each item is a pair of arg-name and arg-value.
  void FeedInArgs(llvm::ArrayRef<std::pair<std::string, ValueRef>> args);

  llvm::ArrayRef<std::string_view> attr_names() const;

  OpExecutableBuilder* NewOpExecutable(std::string_view op_name);
};

}  // namespace cinnrt::host_context

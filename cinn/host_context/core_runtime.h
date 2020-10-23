#pragma once
#include <memory>
#include <string>
#include "llvm/ADT/ArrayRef.h"

namespace cinn::host_context {

class KernelRegistry;
class OpExecutable;
class OpExecutableBuilder;
class SymbolTable;

/**
 * CoreRuntime encapsulate the runtime facilities.
 */
class CoreRuntime {
 public:
  //! Execute a program.
  void Execute();

  //! Get a SymbolTable bound to a function.
  SymbolTable* GetSymbolTable(const std::string& fn_name);

  ~CoreRuntime();

 protected:
  class Impl;
  explicit CoreRuntime(Impl* impl);
  std::unique_ptr<Impl> impl_;
};

/**
 * The builder for CoreRuntime.
 */
class CoreRuntimeBuilder : public CoreRuntime {
 public:
  explicit CoreRuntimeBuilder(KernelRegistry* kernel_registry);

  SymbolTable* NewSymbolTable(std::string_view fn_name);

  llvm::ArrayRef<std::string_view> attr_names() const;

  OpExecutableBuilder* NewOpExecutable(std::string_view op_name, const std::string& fn_name);
};

}  // namespace cinn::host_context

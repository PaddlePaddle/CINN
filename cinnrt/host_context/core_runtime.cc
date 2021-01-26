#include "cinnrt/host_context/core_runtime.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/op_executable.h"
#include "cinnrt/host_context/symbol_table.h"

namespace cinnrt::host_context {

struct CoreRuntime::Impl {
  KernelRegistry* kernel_registry{};
  SymbolTable symbol_table;
  std::vector<OpExecutableBuilder> op_executables;

  std::vector<ValueRef> results;
};

SymbolTable* CoreRuntime::symbol_table() { return &impl_->symbol_table; }

CoreRuntime::CoreRuntime(CoreRuntime::Impl* impl) : impl_(impl) {}

void CoreRuntime::Execute() {
  for (auto& op : impl_->op_executables) {
    op.Execute();
  }
}

CoreRuntimeBuilder::CoreRuntimeBuilder(KernelRegistry* kernel_registry) : CoreRuntime(new Impl) {
  impl_->kernel_registry = kernel_registry ? kernel_registry : GetCpuKernelRegistry();
}

OpExecutableBuilder* CoreRuntimeBuilder::NewOpExecutable(std::string_view op_name, const std::string& fn_name) {
  impl_->op_executables.emplace_back(op_name, symbol_table(), impl_->kernel_registry);
  return &impl_->op_executables.back();
}

void CoreRuntime::FeedInArgs(llvm::ArrayRef<std::pair<std::string, ValueRef>> args) {
  for (auto& item : args) {
    symbol_table()->Register(item.first, item.second);
  }
}

llvm::SmallVector<ValueRef, 4> CoreRuntime::GetResults(llvm::ArrayRef<std::string_view> arg_names) {
  llvm::SmallVector<ValueRef, 4> results;
  for (auto& name : arg_names) {
    results.push_back(ValueRef(symbol_table()->Get(name)));
  }

  return results;
}

CoreRuntime::~CoreRuntime() {}

}  // namespace cinnrt::host_context

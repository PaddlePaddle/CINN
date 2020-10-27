#include "cinn/host_context/core_runtime.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "cinn/host_context/kernel_registry.h"
#include "cinn/host_context/op_executable.h"
#include "cinn/host_context/symbol_table.h"

namespace cinn::host_context {

struct CoreRuntime::Impl {
  KernelRegistry* kernel_registry{};
  std::unordered_map<std::string /*function name*/, SymbolTable> symbol_tables;
  std::vector<OpExecutableBuilder> op_executables;
};

SymbolTable* CoreRuntime::GetSymbolTable(const std::string& fn_name) {
  auto it = impl_->symbol_tables.find(fn_name);
  return it != impl_->symbol_tables.end() ? &it->second : nullptr;
}

CoreRuntime::CoreRuntime(CoreRuntime::Impl* impl) : impl_(impl) {}

void CoreRuntime::Execute() {
  for (auto& op : impl_->op_executables) {
    op.Execute();
  }
}

CoreRuntimeBuilder::CoreRuntimeBuilder(KernelRegistry* kernel_registry) : CoreRuntime(new Impl) {
  impl_->kernel_registry = kernel_registry ? kernel_registry : GetCpuKernelRegistry();
}

SymbolTable* CoreRuntimeBuilder::NewSymbolTable(std::string_view fn_name) {
  return &impl_->symbol_tables.try_emplace(std::string(fn_name)).first->second;
}

OpExecutableBuilder* CoreRuntimeBuilder::NewOpExecutable(std::string_view op_name, const std::string& fn_name) {
  impl_->op_executables.emplace_back(op_name, GetSymbolTable(fn_name), impl_->kernel_registry);
  return &impl_->op_executables.back();
}

CoreRuntime::~CoreRuntime() {}

}  // namespace cinn::host_context

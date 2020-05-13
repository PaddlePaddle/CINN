#include "cinn/backends/llvm/runtime_symbol_registry.h"

#include <string_view>

namespace cinn::backends {

RuntimeSymbolRegistry &RuntimeSymbolRegistry::Global() {
  static RuntimeSymbolRegistry registry;
  return registry;
}

void *RuntimeSymbolRegistry::Lookup(std::string_view name) const {
  std::lock_guard<std::mutex> lock(mu_);

  if (auto it = symbols_.find(std::string(name)); it != symbols_.end()) {
    return it->second;
  }

  return nullptr;
}

void RuntimeSymbolRegistry::Register(const std::string &name, void *address) {
  std::lock_guard<std::mutex> lock(mu_);
  CHECK(address) << "Register a NULL symbol";
  auto it = symbols_.find(name);
  if (it != symbols_.end()) {
    CHECK_EQ(it->second, address) << "Duplicate register symbol [" << name << "]";
    return;
  }

  symbols_.insert({name, reinterpret_cast<void *>(address)});
}
}  // namespace cinn::backends

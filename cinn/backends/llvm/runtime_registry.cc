#include "cinn/backends/llvm/runtime_registry.h"
#include <string_view>

namespace cinn::backends {

/*static*/ RuntimeRegistry *RuntimeRegistry::Global() {
  static RuntimeRegistry registry;
  return &registry;
}

void *RuntimeRegistry::Lookup(std::string_view name) const {
  std::lock_guard<std::mutex> lock(mu_);

  if (auto it = symbols_.find(std::string(name)); it != symbols_.end()) {
    return it->second;
  }

  return nullptr;
}
}  // namespace cinn::backends

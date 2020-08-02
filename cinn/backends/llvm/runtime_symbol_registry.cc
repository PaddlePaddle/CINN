#include "cinn/backends/llvm/runtime_symbol_registry.h"

#include <glog/raw_logging.h>

#include <iostream>
#include <string_view>

namespace cinn {
namespace backends {

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
  RAW_LOG_INFO("JIT Register function [%s]: %p", name.c_str(), address);
  std::lock_guard<std::mutex> lock(mu_);
  auto it = symbols_.find(name);
  if (it != symbols_.end()) {
    CHECK_EQ(it->second, address) << "Duplicate register symbol [" << name << "]";
    return;
  }

  symbols_.insert({name, reinterpret_cast<void *>(address)});
}

}  // namespace backends
}  // namespace cinn

// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/backends/llvm/runtime_symbol_registry.h"

#include <glog/raw_logging.h>

#include <absl/strings/string_view.h>
#include <iostream>

namespace cinn {
namespace backends {

RuntimeSymbolRegistry &RuntimeSymbolRegistry::Global() {
  static RuntimeSymbolRegistry registry;
  return registry;
}

void *RuntimeSymbolRegistry::Lookup(absl::string_view name) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = symbols_.find(std::string(name));
  if (it != symbols_.end()) {
    return it->second;
  }

  return nullptr;
}

void RuntimeSymbolRegistry::Register(const std::string &name, void *address) {
#ifdef CINN_WITH_DEBUG
  RAW_LOG_INFO("JIT Register function [%s]: %p", name.c_str(), address);
#endif  // CINN_WITH_DEBUG
  std::lock_guard<std::mutex> lock(mu_);
  auto it = symbols_.find(name);
  if (it != symbols_.end()) {
    CHECK_EQ(it->second, address) << "Duplicate register symbol [" << name << "]";
    return;
  }

  symbols_.insert({name, reinterpret_cast<void *>(address)});
}

void RuntimeSymbolRegistry::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  symbols_.clear();
  scalar_holder_.clear();
}

}  // namespace backends
}  // namespace cinn

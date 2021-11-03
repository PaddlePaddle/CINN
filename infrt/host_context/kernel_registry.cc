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

#include "infrt/host_context/kernel_registry.h"

#include <absl/container/flat_hash_map.h>

#include "glog/logging.h"
#include "llvm/ADT/SmallVector.h"

namespace infrt {
namespace host_context {

struct KernelRegistry::Impl {
  absl::flat_hash_map<std::string, KernelImplementation> data;
  absl::flat_hash_map<std::string, llvm::SmallVector<std::string, 4>> attr_names;
};

KernelRegistry::KernelRegistry() : impl_(std::make_unique<Impl>()) {}

void KernelRegistry::AddKernel(const std::string &key, KernelImplementation fn) {
  bool added = impl_->data.try_emplace(key, fn).second;
  CHECK(added) << "kernel [" << key << "] is registered twice";
}

void KernelRegistry::AddKernelAttrNameList(const std::string &key, const std::vector<std::string> &names) {
  bool added = impl_->attr_names.try_emplace(key, llvm::SmallVector<std::string, 4>(names.begin(), names.end())).second;
  CHECK(added) << "kernel [" << key << "] is registered twice in attribute names";
}

KernelImplementation KernelRegistry::GetKernel(const std::string &key) const {
  auto it = impl_->data.find(key);
  return it != impl_->data.end() ? it->second : KernelImplementation{};
}

std::vector<std::string> KernelRegistry::GetKernelList() const {
  std::vector<std::string> res(impl_->data.size());
  for (auto i : impl_->data) {
    res.push_back(i.first);
  }
  return res;
}

KernelRegistry::~KernelRegistry() {}

size_t KernelRegistry::size() const { return impl_->data.size(); }

KernelRegistry *GetCpuKernelRegistry() {
  static auto registry = std::make_unique<KernelRegistry>();
  return registry.get();
}

}  // namespace host_context
}  // namespace infrt

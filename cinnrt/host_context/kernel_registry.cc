#include "cinnrt/host_context/kernel_registry.h"

#include <unordered_map>

#include "glog/logging.h"
#include "llvm/ADT/SmallVector.h"

namespace cinnrt {
namespace host_context {

struct KernelRegistry::Impl {
  std::unordered_map<std::string, KernelImplementation> data;
  std::unordered_map<std::string, llvm::SmallVector<std::string, 4>> attr_names;
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

KernelRegistry::~KernelRegistry() {}

KernelRegistry *GetCpuKernelRegistry() {
  static auto registry = std::make_unique<KernelRegistry>();
  return registry.get();
}

}  // namespace host_context
}  // namespace cinnrt

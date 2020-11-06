#include "cinnrt/host_context/kernel_registry.h"

#include <glog/logging.h>

#include <unordered_map>

namespace cinn {
namespace host_context {

struct KernelRegistry::Impl {
  std::unordered_map<std::string, KernelImplementation> data;
  std::unordered_map<std::string_view, llvm::SmallVector<std::string_view, 4>> attr_names;
};

KernelRegistry::KernelRegistry() : impl_(new Impl) {}

void KernelRegistry::AddKernel(std::string_view key, KernelImplementation fn) {
  bool added = impl_->data.try_emplace(std::string(key), fn).second;
  CHECK(added) << "kernel [" << key << "] is registered twice";
}

void KernelRegistry::AddKernelAttrNameList(std::string_view key, llvm::ArrayRef<std::string_view> names) {
  bool added =
      impl_->attr_names.try_emplace(key, llvm::SmallVector<std::string_view, 4>(names.begin(), names.end())).second;
  CHECK(added) << "kernel [" << key << "] is registered twice in attribute names";
}

KernelImplementation KernelRegistry::GetKernel(std::string_view key) const {
  auto it = impl_->data.find(std::string(key));
  return it != impl_->data.end() ? it->second : KernelImplementation{};
}

KernelRegistry::~KernelRegistry() {}

KernelRegistry* GetCpuKernelRegistry() {
  static std::unique_ptr<KernelRegistry> x(new KernelRegistry);
  return x.get();
}

}  // namespace host_context
}  // namespace cinn

#include "cinn/host_context/kernel_registry.h"
#include <glog/logging.h>
#include <unordered_map>

namespace cinn {
namespace host_context {

struct KernelRegistry::Impl {
  std::unordered_map<std::string, KernelImplementation> data;
};

KernelRegistry::KernelRegistry() : impl_(new Impl) {}

void KernelRegistry::AddKernel(std::string_view key, KernelImplementation fn) {
  bool added = impl_->data.try_emplace(std::string(key), fn).second;
  CHECK(added) << "kernel [" << key << "] is registered twice";
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

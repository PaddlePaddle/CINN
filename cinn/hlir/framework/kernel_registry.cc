#include "cinn/hlir/framework/kernel_registry.h"

#include "cinn/hlir/framework/kernel_utils.h"

namespace cinn::hlir::framework {

struct KernelRegistry::Impl {
  std::unordered_map<std::string_view, KernelImplType> impls;
};

KernelRegistry::KernelRegistry() : impl_(new KernelRegistry::Impl) {}

void KernelRegistry::AddKernel(std::string_view name, KernelImplType fn) {
  bool added = impl_->impls.try_emplace(name, fn).second;
  (void)added;
  CHECK(added) << "Re-registered existing kernel";
}

KernelImplType KernelRegistry::GetKernel(std::string_view name) const { return nullptr; }

int add(int a, int b) { return a + b; }

void RegisterIntegerKernels(KernelRegistry *registry) { registry->AddKernel("cinn.contant.i32.add", CINN_KERNEL(add)); }

}  // namespace cinn::hlir::framework

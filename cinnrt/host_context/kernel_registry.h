#pragma once

#include <memory>
#include <string>
#include <vector>

namespace cinnrt {
namespace host_context {

class KernelFrame;

using KernelImplementation = void (*)(KernelFrame *frame);

/**
 * Hold the kernels registered in the system.
 */
class KernelRegistry {
 public:
  KernelRegistry();

  void AddKernel(const std::string &key, KernelImplementation fn);
  void AddKernelAttrNameList(const std::string &key, const std::vector<std::string> &names);

  KernelImplementation GetKernel(const std::string &key) const;
  std::vector<std::string> GetKernelList() const;

  size_t size() const;

  ~KernelRegistry();

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

//! The global CPU kernel registry.
KernelRegistry *GetCpuKernelRegistry();

}  // namespace host_context
}  // namespace cinnrt

/**
 * compile function RegisterKernels in C way to avoid C++ name mangling.
 */
#ifdef __cplusplus
extern "C" {
#endif
void RegisterKernels(cinnrt::host_context::KernelRegistry *registry);
#ifdef __cplusplus
}
#endif

#pragma once
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <memory>

namespace cinnrt {
namespace host_context {

class KernelFrame;

using KernelImplementation = void (*)(KernelFrame* frame);

/**
 * Hold the kernels registered in the system.
 */
class KernelRegistry {
 public:
  KernelRegistry();

  void AddKernel(std::string_view key, KernelImplementation fn);
  void AddKernelAttrNameList(std::string_view key, llvm::ArrayRef<std::string_view> names);

  KernelImplementation GetKernel(std::string_view key) const;

  ~KernelRegistry();

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

//! The global CPU kernel registry.
KernelRegistry* GetCpuKernelRegistry();

}  // namespace host_context
}  // namespace cinnrt

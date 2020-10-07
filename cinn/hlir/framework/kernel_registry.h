#pragma once

#include <memory>
#include <string>

#include "cinn/common/common.h"

namespace cinn::hlir::framework {

class KernelFrame;

using KernelImplType = void (*)(KernelFrame* frame);

/**
 * This is a mapping from the MLIR opcodes to the corresponding functions.
 */
class KernelRegistry {
 public:
  KernelRegistry();

  //! Add a kernel implementation.
  void AddKernel(std::string_view name, KernelImplType fn);

  KernelImplType GetKernel(std::string_view name) const;

  CINN_DISALLOW_COPY_AND_ASSIGN(KernelRegistry);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

void RegisterIntegerKernels(KernelRegistry* registry);

}  // namespace cinn::hlir::framework

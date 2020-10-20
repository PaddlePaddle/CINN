#include "cinn/host_context/mlir_to_runtime_translate.h"
#include <gtest/gtest.h>
#include "cinn/dialect/mlir_loader.h"
#include "cinn/host_context/core_runtime.h"
#include "cinn/host_context/kernel_registry.h"
#include "cinn/host_context/kernel_utils.h"
#include "cinn/kernel/basic_kernels.h"

namespace cinn::host_context {

TEST(MlirToRuntimeTranslate, basic) {
  mlir::MLIRContext context;

  auto source = R"ROC(
func @main() -> () {
  %v0 = cinn.constant.f32 1.0
  %v1 = cinn.constant.f32 2.0
  %v2 = "cinn.add.f32"(%v0, %v1) : (f32, f32) -> f32
  %v3 = "cinn.mul.f32"(%v2, %v1) : (f32, f32) -> f32

  "cinn.print.f32"(%v1) : (f32) -> ()

  cinn.return
}
)ROC";

  auto module = dialect::LoadMlirSource(&context, source);
  module->verify();

  KernelRegistry registry;
  kernel::RegisterFloatBasicKernels(&registry);
  kernel::RegisterIntBasicKernels(&registry);

  CoreRuntimeBuilder core_builder(&registry);
  MlirToRuntimeTranslate(module.get(), &core_builder);

  core_builder.Execute();
}

}  // namespace cinn::host_context
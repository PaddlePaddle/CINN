#include "cinnrt/host_context/mlir_to_runtime_translate.h"

#include <gtest/gtest.h>

#include "cinnrt/common/global.h"
#include "cinnrt/dialect/mlir_loader.h"
#include "cinnrt/host_context/core_runtime.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/kernel_utils.h"
#include "cinnrt/kernel/basic_kernels.h"
#include "cinnrt/kernel/control_flow_kernels.h"
#include "cinnrt/kernel/tensor_kernels.h"
#include "cinnrt/kernel/tensor_shape_kernels.h"
#include "cinnrt/kernel/test_kernels.h"

namespace cinnrt::host_context {

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

  TestMlir(module.get(), &registry);
}

TEST(TestMlir, basic) {
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

  TestMlir(module.get(), &registry);
}

TEST(TestMlir, shadow_copy_tensor_profile) {
  mlir::MLIRContext* context = cinnrt::Global::getMLIRContext();

  auto source = R"ROC(
func @predict(%a: !cinn.tensor<X86, NCHW, F32>, %b: !cinn.tensor<X86, NCHW, F32>) -> (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) {
  %a0 = dt.shallow_copy_tensor %a : !cinn.tensor<X86, NCHW, F32> -> !cinn.tensor<X86, NCHW, F32>
  %b0 = dt.shallow_copy_tensor %b : !cinn.tensor<X86, NCHW, F32> -> !cinn.tensor<X86, NCHW, F32>

  cinn.return %a0, %b0: !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>
}
  )ROC";

  auto module = dialect::LoadMlirSource(context, source);
  module->verify();

  host_context::KernelRegistry registry;

  kernel::RegisterBasicKernels(&registry);
  kernel::RegisterTestKernels(&registry);
  kernel::RegisterTensorShapeKernels(&registry);
  kernel::RegisterTensorKernels(&registry);
  kernel::RegisterControlFlowKernels(&registry);



}

}  // namespace cinnrt::host_context
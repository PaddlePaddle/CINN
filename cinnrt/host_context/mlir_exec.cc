#include <llvm/Support/CommandLine.h>

#include <iostream>
#include <string>

#include "cinnrt/dialect/mlir_loader.h"
#include "cinnrt/host_context/core_runtime.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/mlir_to_runtime_translate.h"
#include "cinnrt/kernel/basic_kernels.h"
#include "cinnrt/kernel/tensor_kernels.h"
#include "cinnrt/kernel/tensor_shape_kernels.h"
#include "cinnrt/common/global.h"

int main(int argc, char** argv) {
  using namespace llvm;    // NOLINT
  using namespace cinnrt;  // NOLINT
  cl::opt<std::string> input_file("i", cl::desc("Specify input filename"), cl::value_desc("input file name"));
  cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext *context = cinnrt::Global::getMLIRContext();
  auto module = dialect::LoadMlirFile(input_file.c_str(), context);

  host_context::KernelRegistry registry;

  kernel::RegisterBasicKernels(&registry);
  kernel::RegisterTensorShapeKernels(&registry);
  kernel::RegisterTensorKernels(&registry);

  host_context::ExecuteMlir(module.get(), &registry);

  std::cout << std::endl;
  return 0;
}

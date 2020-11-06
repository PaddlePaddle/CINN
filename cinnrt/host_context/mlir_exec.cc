#include <llvm/Support/CommandLine.h>

#include <iostream>
#include <string>

#include "cinn/dialect/mlir_loader.h"
#include "cinn/kernel/basic_kernels.h"
#include "cinn/kernel/tensor_kernels.h"
#include "cinn/kernel/tensor_shape_kernels.h"
#include "core_runtime.h"
#include "kernel_registry.h"
#include "mlir_to_runtime_translate.h"

int main(int argc, char** argv) {
  using namespace llvm;  // NOLINT
  using namespace cinn;  // NOLINT
  cl::opt<std::string> input_file("i", cl::desc("Specify input filename"), cl::value_desc("input file name"));
  cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext context;
  auto module = dialect::LoadMlirFile(input_file.c_str(), &context);

  host_context::KernelRegistry registry;

  kernel::RegisterBasicKernels(&registry);
  kernel::RegisterTensorShapeKernels(&registry);
  kernel::RegisterTensorKernels(&registry);

  host_context::ExecuteMlir(module.get(), &registry);

  std::cout << std::endl;
  return 0;
}

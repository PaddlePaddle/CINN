#include <llvm/Support/CommandLine.h>
#include <iostream>
#include <string>
#include "cinn/dialect/mlir_loader.h"
#include "cinn/host_context/core_runtime.h"
#include "cinn/host_context/kernel_registry.h"
#include "cinn/host_context/mlir_to_runtime_translate.h"
#include "cinn/kernel/basic_kernels.h"

int main(int argc, char** argv) {
  using namespace llvm;  // NOLINT
  using namespace cinn;  // NOLINT
  cl::opt<std::string> input_file("i", cl::desc("Specify input filename"), cl::value_desc("input file name"));
  cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext context;
  auto module = dialect::LoadMlirFile(input_file.c_str(), &context);

  host_context::KernelRegistry registry;

  kernel::RegisterBasicKernels(&registry);

  host_context::ExecuteMlir(module.get(), &registry);

  std::cout << std::endl;
  return 0;
}

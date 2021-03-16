#include <llvm/Support/CommandLine.h>

#include <iostream>
#include <string>

#include "cinnrt/common/global.h"
#include "cinnrt/dialect/mlir_loader.h"
#include "cinnrt/host_context/core_runtime.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/mlir_to_runtime_translate.h"
#include "cinnrt/kernel/basic_kernels.h"
#include "cinnrt/kernel/control_flow_kernels.h"
#include "cinnrt/kernel/tensor_kernels.h"
#include "cinnrt/kernel/tensor_shape_kernels.h"
#include "cinnrt/kernel/test_kernels.h"
#include "llvm/Support/DynamicLibrary.h"

static llvm::cl::list<std::string> cl_shared_libs(  // NOLINT
    "shared_libs",
    llvm::cl::desc("Specify shared library with kernels."),
    llvm::cl::ZeroOrMore,
    llvm::cl::MiscFlags::CommaSeparated);

int main(int argc, char** argv) {
  using namespace llvm;    // NOLINT
  using namespace cinnrt;  // NOLINT
  cl::opt<std::string> input_file("i", cl::desc("Specify input filename"), cl::value_desc("input file name"));
  cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext* context = cinnrt::Global::getMLIRContext();
  auto module                = dialect::LoadMlirFile(input_file.c_str(), context);

  host_context::KernelRegistry registry;

  kernel::RegisterBasicKernels(&registry);
  kernel::RegisterTestKernels(&registry);
  kernel::RegisterTensorShapeKernels(&registry);
  kernel::RegisterTensorKernels(&registry);
  kernel::RegisterControlFlowKernels(&registry);

  // load extra shared library
  for (const auto& lib_path : cl_shared_libs) {
    std::string err;
    llvm::sys::DynamicLibrary dynLib = llvm::sys::DynamicLibrary::getPermanentLibrary(lib_path.c_str(), &err);
    if (!dynLib.isValid()) {
      llvm::errs() << "Load shared library failed. Error: " << err << "\n";
      return 1;
    }
    if (auto reg_sym = dynLib.SearchForAddressOfSymbol("RegisterKernels")) {
      auto reg_func = reinterpret_cast<void (*)(host_context::KernelRegistry*)>(reg_sym);
      reg_func(&registry);
    } else {
      llvm::outs() << "Symbol \"RegisterKernels\" not found in \"" << lib_path << "\". Skip.\n";
    }
  }

  host_context::TestMlir(module.get(), &registry);

  std::cout << std::endl;
  return 0;
}

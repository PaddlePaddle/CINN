
#include <glog/logging.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <iostream>

#include "cinnrt/common/global.h"
#include "cinnrt/dialect/init_cinn_dialects.h"
#include "cinnrt/dialect/mlir_loader.h"

int main(int argc, char **argv) {
  mlir::MLIRContext *context = cinnrt::Global::getMLIRContext();
  auto &registry             = context->getDialectRegistry();
  cinnrt::RegisterCinnDialects(registry);

  llvm::cl::opt<std::string> input_file("i", llvm::cl::desc("Specify input filename"), llvm::cl::value_desc("input file name"));
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto module = cinnrt::dialect::LoadMlirFile(input_file.c_str(), context);

  mlir::PassManager passManager(context);
  passManager.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  if (mlir::failed(passManager.run(*module))) {
    std::cerr << "passManager run failed." << std::endl;
    return 4;
  }
  module->dump();
  return mlir::failed(mlir::MlirOptMain(argc, argv, "CINN mlir pass driver", registry, true));
}

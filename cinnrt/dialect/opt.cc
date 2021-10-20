#include <glog/logging.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <iostream>

#include "cinnrt/common/global.h"
#include "cinnrt/dialect/init_cinn_dialects.h"
#include "cinnrt/dialect/mlir_loader.h"

int main(int argc, char **argv) {
  mlir::MLIRContext *context = cinnrt::Global::getMLIRContext();

  auto &registry = context->getDialectRegistry();
  cinnrt::RegisterCinnDialects(registry);

  mlir::registerCanonicalizerPass();

  return mlir::failed(mlir::MlirOptMain(argc, argv, "CINN mlir pass driver", registry));
}

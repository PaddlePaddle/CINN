#include "cinn/dialect/mlir_loader.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/Function.h>
#include <mlir/Parser.h>

#include <string>

#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/dialect/init_cinn_dialects.h"

namespace cinn::dialect {

TEST(MlirLoader, basic) {
  mlir::MLIRContext context;

  auto source = R"ROC(
func @main() -> f32 {
  %v0 = cinn.constant.f32 1.0
  %v1 = cinn.constant.f32 2.0
  %value = "cinn.add.f32"(%v0, %v1) : (f32, f32) -> f32

  "cinn.print.f32"(%v0) : (f32) -> ()

  cinn.return %value : f32
}
)ROC";

  auto module = LoadMlirSource(&context, source);
  module->verify();

  LOG(INFO) << "module name: " << module->getOperationName().data();
  for (auto func : module->getOps<mlir::FuncOp>()) {
    LOG(INFO) << "get func " << func.getName().str();
  }

  MlirToFrontend(module.release());
}

}  // namespace cinn::dialect

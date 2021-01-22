#include <glog/logging.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Support/MlirOptMain.h>

#include "cinnrt/dialect/init_cinn_dialects.h"
#include "cinnrt/common/global.h"

int main(int argc, char** argv) {
  mlir::MLIRContext *context = cinnrt::Global::getMLIRContext();
  context->allowUnregisteredDialects();
  auto &registry = context->getDialectRegistry();
  cinnrt::RegisterCinnDialects(registry);
  //context->getOrLoadDialect<test::TestDialect>();
  return mlir::failed(mlir::MlirOptMain(argc, argv, "CINN mlir pass driver", registry, true));
}

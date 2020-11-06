#include <glog/logging.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Support/MlirOptMain.h>

#include "cinnrt/dialect/init_cinn_dialects.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  cinnrt::RegisterCinnDialects(registry);
  return mlir::failed(mlir::MlirOptMain(argc, argv, "CINN", registry, true));
}

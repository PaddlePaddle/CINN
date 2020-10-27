#pragma once

#include <mlir/IR/Module.h>
#include <memory>
#include <string_view>
#include "cinn/frontend/syntax.h"

namespace cinn::dialect {

mlir::OwningModuleRef LoadMlirSource(mlir::MLIRContext* context, std::string_view mlir_source);
mlir::OwningModuleRef LoadMlirFile(std::string_view file_name, mlir::MLIRContext* context);

std::unique_ptr<frontend::Program> MlirToFrontend(mlir::ModuleOp module);

}  // namespace cinn::dialect

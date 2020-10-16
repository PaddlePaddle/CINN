#pragma once

#include <mlir/IR/Module.h>
#include <memory>
#include <string_view>
#include "cinn/frontend/syntax.h"

namespace cinn::dialect {

mlir::OwningModuleRef LoadMlirSource(mlir::MLIRContext* context, std::string_view mlir_source);

std::unique_ptr<frontend::Program> MlirToFrontend(mlir::ModuleOp module);

}  // namespace cinn::dialect

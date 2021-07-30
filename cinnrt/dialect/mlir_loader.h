#pragma once

#include <mlir/IR/Module.h>

#include <memory>
#include <string_view>

#include "cinnrt/paddle/syntax.h"

namespace cinnrt::dialect {

mlir::OwningModuleRef LoadMlirSource(mlir::MLIRContext* context, std::string_view mlir_source);
mlir::OwningModuleRef LoadMlirFile(std::string_view file_name, mlir::MLIRContext* context);

std::unique_ptr<cinnrt::paddle::Program> MlirToFrontend(mlir::ModuleOp module);

}  // namespace cinnrt::dialect

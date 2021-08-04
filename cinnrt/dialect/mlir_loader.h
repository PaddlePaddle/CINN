#pragma once

#include <glog/logging.h>

#include <mlir/IR/Module.h>

#include <memory>
#include <string_view>

namespace cinnrt::dialect {

mlir::OwningModuleRef LoadMlirSource(mlir::MLIRContext* context, std::string_view mlir_source);
mlir::OwningModuleRef LoadMlirFile(std::string_view file_name, mlir::MLIRContext* context);

}  // namespace cinnrt::dialect

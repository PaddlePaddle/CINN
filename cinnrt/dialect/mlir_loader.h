#pragma once

#include <glog/logging.h>

#include <mlir/IR/Module.h>

#include <memory>
#include "absl/strings/string_view.h"

namespace cinnrt::dialect {

mlir::OwningModuleRef LoadMlirSource(mlir::MLIRContext* context, absl::string_view mlir_source);
mlir::OwningModuleRef LoadMlirFile(absl::string_view file_name, mlir::MLIRContext* context);

}  // namespace cinnrt::dialect

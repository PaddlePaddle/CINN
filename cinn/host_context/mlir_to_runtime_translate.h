#pragma once

#include <mlir/IR/Module.h>

namespace cinn::host_context {

class CoreRuntimeBuilder;

/**
 * Build a CoreRuntime from a MLIR module.
 */
void MlirToRuntimeTranslate(mlir::ModuleOp module, cinn::host_context::CoreRuntimeBuilder *runtime);

}  // namespace cinn::host_context

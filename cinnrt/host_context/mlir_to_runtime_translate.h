#pragma once

#include <mlir/IR/Module.h>

namespace cinnrt::host_context {

class CoreRuntimeBuilder;
class Value;
class KernelRegistry;

class MlirToRuntimeTranslator {
 public:
  MlirToRuntimeTranslator(mlir::ModuleOp module, CoreRuntimeBuilder* runtime);

  void Emit();

  //! Emit a "cinn.constant.*" operation, return true if succeed.
  bool EmitConstantOp(mlir::Operation* op);
  //! Emit a "cinn.return" operation.
  bool EmitReturnOp(mlir::Operation* op);
  //! Emit a "ts.build_shape" operation.
  bool EmitBuildShapeOp(mlir::Operation* op);
  //! Emit an operation other than the special cases above.
  bool EmitGeneralOp(mlir::Operation* op);

  bool EmitFunctionDef(mlir::Operation* op);

  bool EmitCallOp(mlir::Operation* op);

  template <typename T>
  std::optional<T> EmitAttribute(const mlir::Attribute* attr);

  Value* GetOpResult(mlir::Operation* op);

  Value* GetValue(mlir::Value value);

  Value* AddValue(mlir::Value value);

  void UpdateCurFuncName(std::string_view name);

  ~MlirToRuntimeTranslator();

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * Build a CoreRuntime from a MLIR module.
 */
void MlirToRuntimeTranslate(mlir::ModuleOp module, CoreRuntimeBuilder* runtime);

void ExecuteMlir(mlir::ModuleOp module, KernelRegistry* registry);

}  // namespace cinnrt::host_context

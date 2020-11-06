#pragma once
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/Diagnostics.h>
#include <memory>

namespace cinn::dialect {

/**
 * A scoped diagnostic handler to help debug MLIR process.
 */
class MyScopedDiagnosicHandler : public mlir::SourceMgrDiagnosticHandler {
 public:
  MyScopedDiagnosicHandler(mlir::MLIRContext* ctx, bool propagate);

  mlir::LogicalResult handler(mlir::Diagnostic* diag);

  ~MyScopedDiagnosicHandler();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cinn::dialect

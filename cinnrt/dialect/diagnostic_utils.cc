#include "diagnostic_utils.h"

#include <string>

namespace cinn::dialect {

struct MyScopedDiagnosicHandler::Impl {
  Impl() : diag_stream_(diag_str_) {}

  // String stream to assemble the final error message.
  std::string diag_str_;
  llvm::raw_string_ostream diag_stream_;

  // A SourceMgr to use for the base handler class.
  llvm::SourceMgr source_mgr_;

  // Log detail information.
  bool log_info_{};
};

MyScopedDiagnosicHandler::MyScopedDiagnosicHandler(mlir::MLIRContext *ctx, bool propagate)
    : impl_(new Impl), mlir::SourceMgrDiagnosticHandler(impl_->source_mgr_, ctx, impl_->diag_stream_) {
  setHandler([this](mlir::Diagnostic &diag) { return this->handler(&diag); });
}

mlir::LogicalResult MyScopedDiagnosicHandler::handler(mlir::Diagnostic *diag) {
  if (diag->getSeverity() != mlir::DiagnosticSeverity::Error && !impl_->log_info_) return mlir::success();
  emitDiagnostic(*diag);
  impl_->diag_stream_.flush();
  return mlir::failure(true);
}

}  // namespace cinn::dialect

#include "infrt/dialect/mlir_loader.h"

#include <absl/container/flat_hash_map.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "infrt/dialect/diagnostic_utils.h"
#include "infrt/dialect/init_cinn_dialects.h"

namespace infrt::dialect {

mlir::OwningModuleRef LoadMlirSource(mlir::MLIRContext* context, absl::string_view mlir_source) {
  context->allowUnregisteredDialects();
  RegisterCinnDialects(context->getDialectRegistry());
  context->getDialectRegistry().insert<mlir::StandardOpsDialect>();

  mlir::ScopedDiagnosticHandler scope_handler(context, [](mlir::Diagnostic& diag) {
    if (diag.getSeverity() != mlir::DiagnosticSeverity::Error) return mlir::success();
    VLOG(1) << "diag: " << diag.str();
    return mlir::failure(true);
  });

  auto res = mlir::parseSourceString(llvm::StringRef(mlir_source.data(), mlir_source.length()), context);
  CHECK(*res) << "failed to parse MLIR string";
  return res;
}

mlir::OwningModuleRef LoadMlirFile(absl::string_view file_name, mlir::MLIRContext* context) {
  context->allowUnregisteredDialects();
  RegisterCinnDialects(context->getDialectRegistry());
  context->getDialectRegistry().insert<mlir::StandardOpsDialect>();

  mlir::ScopedDiagnosticHandler scope_handler(context, [](mlir::Diagnostic& diag) {
    if (diag.getSeverity() != mlir::DiagnosticSeverity::Error) return mlir::success();
    VLOG(1) << "diag: " << diag.str();
    return mlir::failure(true);
  });

  return mlir::parseSourceFile(std::string(file_name), context);
}

class Translator {
 public:
  explicit Translator(mlir::ModuleOp module) : module_(module) {}

  void Build() {
    std::vector<std::pair<std::string, mlir::Region*>> named_regions;
    named_regions.reserve(std::distance(module_.begin(), module_.end()));

    int subgraph_idx                             = 0;
    mlir::FuncOp main_fn                         = module_.lookupSymbol<mlir::FuncOp>("main");
    subgraph_index_map_[main_fn.getName().str()] = subgraph_idx++;
    named_regions.emplace_back("main", &main_fn.getBody());

    CHECK_EQ(named_regions.size(), 1UL) << "CINN not support subgraphs yet.";
    for (auto& region : named_regions) {
      BuildSubGraph(region.first, region.second);
    }
  }

 private:
  void BuildSubGraph(const std::string& name, mlir::Region* region) {
    VLOG(1) << "building subgraph [" << name << "]";
    auto& bb = region->front();

    for (unsigned i = 0, e = bb.getNumArguments(); i < e; ++i) {
      mlir::BlockArgument arg = bb.getArgument(i);
      std::string name        = "arg" + std::to_string(i);
    }

    for (auto& inst : bb) {
      if (inst.isKnownTerminator()) break;
      for (auto val : inst.getResults()) {
        VLOG(1) << "get instruction: " << inst.getName().getStringRef().str();
        for (auto& op : inst.getOpOperands()) {
          VLOG(1) << "operand owner: " << op.getOwner()->getName().getStringRef().str();
          VLOG(1) << "op " << op.getOperandNumber();
        }
      }
    }
  }

  mlir::ModuleOp module_;
  absl::flat_hash_map<std::string, int> subgraph_index_map_;
};

}  // namespace infrt::dialect

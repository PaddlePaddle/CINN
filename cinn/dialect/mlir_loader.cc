#include "cinn/dialect/mlir_loader.h"

#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cinn/dialect/diagnostic_utils.h"
#include "cinn/dialect/init_cinn_dialects.h"
#include "cinn/frontend/syntax.h"

namespace cinn::dialect {

mlir::OwningModuleRef LoadMlirSource(mlir::MLIRContext* context, std::string_view mlir_source) {
  context->allowUnregisteredDialects();
  RegisterCinnDialects(context->getDialectRegistry());
  context->getDialectRegistry().insert<mlir::StandardOpsDialect>();

  mlir::ScopedDiagnosticHandler scope_handler(context, [](mlir::Diagnostic& diag) {
    if (diag.getSeverity() != mlir::DiagnosticSeverity::Error) return mlir::success();
    LOG(INFO) << "diag: " << diag.str();
    return mlir::failure(true);
  });

  auto res = mlir::parseSourceString(mlir_source, context);
  CHECK(*res) << "failed to parse MLIR string";
  return res;
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
    LOG(INFO) << "building subgraph [" << name << "]";
    auto& bb = region->front();

    for (unsigned i = 0, e = bb.getNumArguments(); i < e; ++i) {
      mlir::BlockArgument arg = bb.getArgument(i);
      std::string name        = "arg" + std::to_string(i);
    }

    for (auto& inst : bb) {
      if (inst.isKnownTerminator()) break;
      for (auto val : inst.getResults()) {
        LOG(INFO) << "get instruction: " << inst.getName().getStringRef().str();
        for (auto& op : inst.getOpOperands()) {
          LOG(INFO) << "operand owner: " << op.getOwner()->getName().getStringRef().str();
          LOG(INFO) << "op " << op.getOperandNumber();
        }
      }
    }
  }

  mlir::ModuleOp module_;
  std::unordered_map<std::string, int> subgraph_index_map_;
};

std::unique_ptr<frontend::Program> MlirToFrontend(mlir::ModuleOp module) {
  Translator translator(module);
  translator.Build();
  return std::unique_ptr<frontend::Program>();
}

}  // namespace cinn::dialect

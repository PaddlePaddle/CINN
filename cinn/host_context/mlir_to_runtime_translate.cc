#include "cinn/host_context/mlir_to_runtime_translate.h"
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
#include "cinn/dialect/mlir_loader.h"
#include "cinn/host_context/core_runtime.h"
#include "cinn/host_context/op_executable.h"
#include "cinn/host_context/value.h"

namespace cinn::host_context {

template <typename T>
std::string DumpToString(T& op) {  // NOLINT
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op.print(os);
  os.flush();
  return buffer;
}

class Translator {
 public:
  Translator(mlir::ModuleOp module, CoreRuntimeBuilder* runtime) : module_(module), runtime_(runtime) {}

  void Emit() {
    auto main_fn = module_.lookupSymbol<mlir::FuncOp>("main");
    CHECK(main_fn) << "need main function as entry point of the whole program";
    CHECK_EQ(main_fn.getNumArguments(), 0) << "main function not support input arguments";
    UpdateCurFuncName("main");

    auto& block = main_fn.front();

    for (auto& op : block) {
      VLOG(3) << "instr: " << DumpToString(op);

      if (EmitConstant(&op)) continue;
      if (EmitReturn(&op)) continue;
      if (EmitGeneralOp(&op)) continue;
      LOG(FATAL) << "failed to emit op: " << DumpToString(op);
    }
  }

  //! Emit a "cinn.constant.*" operation, return true if succeed.
  bool EmitConstant(mlir::Operation* op);
  //! Emit a "cinn.return" operation.
  bool EmitReturn(mlir::Operation* op);
  //! Emit an operation other than the special cases above.
  bool EmitGeneralOp(mlir::Operation* op);

  Value* GetOpResult(mlir::Operation* op) {
    auto it = op_results_.find(op);
    return it == op_results_.end() ? nullptr : it->second.front().get();
  }

  Value* GetValue(mlir::Value value) {
    auto it = value_map_.find(value);
    return it == value_map_.end() ? nullptr : it->second.get();
  }

  Value* AddValue(mlir::Value value) {
    auto res = value_map_.try_emplace(value, ValueRef(new Value));
    CHECK(res.second) << "Duplicate add mlir value [" << DumpToString(value) << "]";
    return res.first->second.get();
  }

  void UpdateCurFuncName(std::string_view name) { cur_func_name_ = name; }

 private:
  mlir::ModuleOp module_;
  CoreRuntimeBuilder* runtime_{};
  OpExecutableBuilder* cur_op_{};

  // record the current function name.
  std::string cur_func_name_;

  // Map from an operation to its results.
  std::unordered_map<const mlir::Operation*, std::vector<ValueRef>> op_results_;
  llvm::DenseMap<mlir::Value, ValueRef> value_map_;
};

bool Translator::EmitConstant(mlir::Operation* op) {
  if (!utils::Startswith(op->getName().getStringRef().str(), "cinn.constant")) return false;
  VLOG(3) << "Emitting constant op [" << op->getName().getStringRef().str() << "]";

  auto attr = op->getAttr("value");
  if (attr.isa<mlir::FloatAttr>()) {
    if (attr.getType().isF32()) {
      op_results_[op] = {ValueRef(static_cast<float>(attr.cast<mlir::FloatAttr>().getValueAsDouble()))};
    } else if (attr.getType().isF64()) {
      op_results_[op] = {ValueRef(static_cast<double>(attr.cast<mlir::FloatAttr>().getValueAsDouble()))};
    } else {
      LOG(FATAL) << "Not supported attribute type";
    }
    return true;
  }

  if (attr.isa<mlir::IntegerAttr>()) {
    if (attr.getType().isInteger(32)) {
      op_results_[op] = {ValueRef(static_cast<int32_t>(attr.cast<mlir::IntegerAttr>().getSInt()))};
    } else if (attr.getType().isInteger(64)) {
      op_results_[op] = {ValueRef(static_cast<int64_t>(attr.cast<mlir::IntegerAttr>().getSInt()))};
    } else if (attr.getType().isInteger(1)) {
      op_results_[op] = {ValueRef(static_cast<bool>(attr.cast<mlir::IntegerAttr>().getInt()))};
    } else {
      LOG(FATAL) << "Not supported attribute type";
    }
    return true;
  }

  LOG(FATAL) << "Not supported constant attribute type";
  return true;
}

bool Translator::EmitGeneralOp(mlir::Operation* op) {
  cur_op_ = runtime_->NewOpExecutable(op->getName().getStringRef().str(), cur_func_name_);

  // process operands
  for (int i = 0, e = op->getNumOperands(); i < e; i++) {
    auto operand = op->getOperand(i);
    if (operand.getKind() == mlir::Value::Kind::BlockArgument) LOG(FATAL) << "Not support BlockArgument";
    Value* arg_value = GetValue(operand);
    if (!arg_value) {
      auto upstream_op = operand.getDefiningOp();
      arg_value        = GetOpResult(upstream_op);
    }
    CHECK(arg_value) << "No-exist argument value found: " << DumpToString(operand);
    cur_op_->AppendArgument(arg_value);
  }

  // process results
  llvm::SmallVector<Value*, 4> res_values;
  for (int i = 0, e = op->getNumResults(); i < e; i++) {
    auto res = op->getResult(i);
    res_values.push_back(AddValue(res));
  }
  cur_op_->SetResults(res_values);

  // process attributes
  CHECK(op->getAttrs().empty()) << "Not support attribute yet";
  return true;
}

bool Translator::EmitReturn(mlir::Operation* op) {
  if (op->getName().getStringRef() == "cinn.return") return true;
  return false;
}

void MlirToRuntimeTranslate(mlir::ModuleOp module, CoreRuntimeBuilder* runtime) {
  mlir::MLIRContext* ctx = module.getContext();
  Translator(module, runtime).Emit();
}

}  // namespace cinn::host_context

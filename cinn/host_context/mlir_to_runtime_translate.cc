#include "cinn/host_context/mlir_to_runtime_translate.h"
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "cinn/dialect/mlir_loader.h"
#include "cinn/dialect/tensor_shape.h"
#include "cinn/host_context/core_runtime.h"
#include "cinn/host_context/kernel_registry.h"
#include "cinn/host_context/op_executable.h"
#include "cinn/host_context/tensor_shape.h"
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

struct MlirToRuntimeTranslator::Impl {
  mlir::ModuleOp module;
  CoreRuntimeBuilder* runtime{};
  OpExecutableBuilder* cur_op{};

  // record the current function name.
  std::string cur_func_name;

  // Map from an operation to its results.
  std::unordered_map<const mlir::Operation*, std::vector<ValueRef>> op_results;
  llvm::DenseMap<mlir::Value, ValueRef> value_map;
};

bool MlirToRuntimeTranslator::EmitConstantOp(mlir::Operation* op) {
  if (!utils::Startswith(op->getName().getStringRef().str(), "cinn.constant")) return false;
  VLOG(3) << "Emitting constant op [" << op->getName().getStringRef().str() << "]";

  auto attr = op->getAttr("value");
  if (attr.isa<mlir::FloatAttr>()) {
    if (attr.getType().isF32()) {
      impl_->op_results[op] = {ValueRef(static_cast<float>(attr.cast<mlir::FloatAttr>().getValueAsDouble()))};
    } else if (attr.getType().isF64()) {
      impl_->op_results[op] = {ValueRef(static_cast<double>(attr.cast<mlir::FloatAttr>().getValueAsDouble()))};
    } else {
      LOG(FATAL) << "Not supported attribute type";
    }
    return true;
  }

  if (attr.isa<mlir::IntegerAttr>()) {
    if (attr.getType().isInteger(32)) {
      impl_->op_results[op] = {ValueRef(static_cast<int32_t>(attr.cast<mlir::IntegerAttr>().getSInt()))};
    } else if (attr.getType().isInteger(64)) {
      impl_->op_results[op] = {ValueRef(static_cast<int64_t>(attr.cast<mlir::IntegerAttr>().getSInt()))};
    } else if (attr.getType().isInteger(1)) {
      impl_->op_results[op] = {ValueRef(static_cast<bool>(attr.cast<mlir::IntegerAttr>().getInt()))};
    } else {
      LOG(FATAL) << "Not supported attribute type";
    }
    return true;
  }

  LOG(FATAL) << "Not supported constant attribute type";
  return true;
}

bool MlirToRuntimeTranslator::EmitGeneralOp(mlir::Operation* op) {
  impl_->cur_op = impl_->runtime->NewOpExecutable(op->getName().getStringRef().str(), impl_->cur_func_name);

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
    impl_->cur_op->AppendArgument(arg_value);
  }

  // process results
  llvm::SmallVector<Value*, 4> res_values;
  for (int i = 0, e = op->getNumResults(); i < e; i++) {
    auto res = op->getResult(i);
    res_values.push_back(AddValue(res));
  }
  impl_->cur_op->SetResults(res_values);

  // process attributes
  auto attrs = op->getAttrs();
  for (int i = 0; i < attrs.size(); i++) {
    auto& attr = attrs[i];
  }

  CHECK(op->getAttrs().empty()) << "Not support attribute yet";
  return true;
}

bool MlirToRuntimeTranslator::EmitReturnOp(mlir::Operation* op) {
  if (op->getName().getStringRef() == "cinn.return") return true;
  return false;
}

void MlirToRuntimeTranslator::Emit() {
  auto main_fn = impl_->module.lookupSymbol<mlir::FuncOp>("main");
  CHECK(main_fn) << "need main function as entry point of the whole program";
  CHECK_EQ(main_fn.getNumArguments(), 0) << "main function not support input arguments";
  UpdateCurFuncName("main");

  auto& block = main_fn.front();

  for (auto& op : block) {
    VLOG(3) << "instr: " << DumpToString(op);

    if (EmitConstantOp(&op)) continue;
    if (EmitBuildShapeOp(&op)) continue;
    if (EmitReturnOp(&op)) continue;
    if (EmitGeneralOp(&op)) continue;
    LOG(FATAL) << "failed to emit op: " << DumpToString(op);
  }
}

Value* MlirToRuntimeTranslator::GetOpResult(mlir::Operation* op) {
  auto it = impl_->op_results.find(op);
  return it == impl_->op_results.end() ? nullptr : it->second.front().get();
}

Value* MlirToRuntimeTranslator::GetValue(mlir::Value value) {
  auto it = impl_->value_map.find(value);
  return it == impl_->value_map.end() ? nullptr : it->second.get();
}

Value* MlirToRuntimeTranslator::AddValue(mlir::Value value) {
  auto res = impl_->value_map.try_emplace(value, ValueRef(new Value));
  CHECK(res.second) << "Duplicate add mlir value [" << DumpToString(value) << "]";
  return res.first->second.get();
}

MlirToRuntimeTranslator::~MlirToRuntimeTranslator() {}

void MlirToRuntimeTranslator::UpdateCurFuncName(std::string_view name) { impl_->cur_func_name = name; }

MlirToRuntimeTranslator::MlirToRuntimeTranslator(mlir::ModuleOp module, CoreRuntimeBuilder* runtime) : impl_(new Impl) {
  impl_->module  = module;
  impl_->runtime = runtime;
}

bool MlirToRuntimeTranslator::EmitBuildShapeOp(mlir::Operation* op) {
  LOG(INFO) << "processing build shape";
  if (op->getName().getStringRef() != "ts.build_shape") return false;

  auto value = op->getAttr("value");

  CHECK(value.isa<mlir::ArrayAttr>());
  auto values = value.cast<mlir::ArrayAttr>().getValue();
  std::vector<int64_t> dims;
  for (auto& attr_v : values) {
    dims.push_back(attr_v.cast<mlir::IntegerAttr>().getInt());
  }
  impl_->op_results[op] = {ValueRef(new Value(TensorShape(llvm::ArrayRef<int64_t>(dims))))};

  return true;
}

void MlirToRuntimeTranslate(mlir::ModuleOp module, CoreRuntimeBuilder* runtime) {
  mlir::MLIRContext* ctx = module.getContext();
  MlirToRuntimeTranslator(module, runtime).Emit();
}

class FunctionExecute : public MlirToRuntimeTranslator {
 public:
  FunctionExecute(mlir::ModuleOp module, KernelRegistry* registry)
      : MlirToRuntimeTranslator(module, nullptr), registry(registry) {
    CHECK(registry);
  }

  void Emit() {
    CHECK(registry);
    CoreRuntimeBuilder runtime(registry);
    for (auto func_op : impl_->module.getOps<mlir::FuncOp>()) {
      EmitAndRunFunc(func_op);
    }
  }

 private:
  void EmitAndRunFunc(mlir::FuncOp func) {
    // print the function name for llvm FileChecker macro, CHECK-LABEL
    std::cout << func.getName().str() << std::endl;
    if (func.getNumArguments() == 0) {  // an entry function, execute it immediately
      // Emit and execute each function
      CoreRuntimeBuilder runtime(registry);
      impl_->runtime = &runtime;

      auto& blocks = func.getBlocks();
      CHECK_EQ(blocks.size(), 1UL) << "function with more than one block is not supported yet";

      for (auto& op : blocks.front()) {
        if (EmitConstantOp(&op)) continue;
        if (EmitBuildShapeOp(&op)) continue;
        if (EmitReturnOp(&op)) continue;
        if (EmitGeneralOp(&op)) continue;
        LOG(FATAL) << "Not supported op: " << DumpToString(op);
      }

      runtime.Execute();

    } else {
      LOG(FATAL) << "Callable function is not supported yet";
    }
  }

 private:
  KernelRegistry* registry{};
};

void ExecuteMlir(mlir::ModuleOp module, KernelRegistry* registry) {
  FunctionExecute execute(module, registry);
  execute.Emit();
}

}  // namespace cinn::host_context

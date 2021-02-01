#include "cinnrt/host_context/mlir_function_executable.h"

#include <glog/logging.h>

#include "cinnrt/host_context/core_runtime.h"

namespace cinnrt {
namespace host_context {

template <typename T>
std::string DumpToString(T& op) {  // NOLINT
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op.print(os);
  os.flush();
  return buffer;
}

MlirFunctionExecutable::MlirFunctionExecutable(mlir::FuncOp func_op,
                                               CoreRuntimeBuilder* core_runtime_builder,
                                               MlirToRuntimeTranslator::function_defs_t& function_table)
    : Function(func_op.getName().str(), func_op.getNumArguments(), func_op.getNumResults()),
      func_op_(func_op),
      core_runtime_builder_(core_runtime_builder),
      function_table_(function_table),
      MlirToRuntimeTranslator(core_runtime_builder) {
  VLOG(3) << "MlirFunction building function " << func_op.getName().str();
  CHECK(core_runtime_builder_);
}

MlirFunctionExecutable::MlirFunctionExecutable(mlir::FuncOp func_op,
                                               KernelRegistry* kernel_registry,
                                               MlirToRuntimeTranslator::function_defs_t& function_table)
    : Function(func_op.getName().str(), func_op.getNumArguments(), func_op.getNumResults()),
      core_runtime_builder_(new CoreRuntimeBuilder(kernel_registry)),
      function_table_(function_table),
      MlirToRuntimeTranslator(core_runtime_builder_.get()) {}

void MlirFunctionExecutable::BuildExecutables(llvm::ArrayRef<Value*> arguments,
                                              llvm::MutableArrayRef<ValueRef> results) {
  CHECK_EQ(arguments.size(), func_op_.getNumArguments());
  // We use the function call's arguments as op_executable's operands to avoid copy.
  for (int i = 0; i < func_op_.getNumArguments(); i++) {
    AddValue(func_op_.getArgument(i), arguments[i]);
  }

  // build the program
  auto& blocks = func_op_.getBlocks();
  CHECK_EQ(blocks.size(), 1UL) << "function with more than one block is not supported yet";

  llvm::SmallVector<Value*, 3> runtime_results;
  for (auto& op : blocks.front()) {
    if (EmitConstantOp(&op)) continue;
    if (EmitBuildShapeOp(&op)) continue;

    llvm::SmallVector<mlir::Value, 3> mlir_results;
    if (EmitReturnOp(&op, &mlir_results)) {
      for (auto v : mlir_results) {
        runtime_results.push_back(GetValue(v));
      }
      continue;
    }

    if (EmitCallOp(&op, &function_table_)) continue;

    if (EmitGeneralOp(&op)) continue;
    LOG(FATAL) << "Not supported op: " << DumpToString(op);
  }

  // after the block is built, we can get the result values of the whole function call in the runtime_resutls.

  mlir::SmallVector<Value*, 3> results_copied;
  for (ValueRef& x : results) {
    results_copied.push_back(x.get());
  }

  // set a lambda function to help copy the results from the runtime results in the local function to outer program.
  CHECK_EQ(results_copied.size(), runtime_results.size());
  this->copy_res_fn_ = [results_copied, runtime_results] {
    VLOG(4) << "copy results to result";
    for (int i = 0; i < results_copied.size(); i++) {
      VLOG(4) << ".. copy " << runtime_results[i] << " to " << results_copied[i];
      CopyTo(*runtime_results[i], results_copied[i]);
    }
  };
}

void MlirFunctionExecutable::Execute(llvm::ArrayRef<Value*> arguments, llvm::MutableArrayRef<ValueRef> results) const {
  CHECK_EQ(arguments.size(), num_arguments());
  CHECK_EQ(results.size(), num_results());

  if (core_runtime_builder_->num_ops() == 0) {
    const_cast<MlirFunctionExecutable*>(this)->BuildExecutables(arguments, results);
  }

  auto& func_op = *const_cast<mlir::FuncOp*>(&func_op_);

  const_cast<CoreRuntimeBuilder*>(core_runtime_builder_.get())->Execute();

  copy_res_fn_();
}

}  // namespace host_context
}  // namespace cinnrt

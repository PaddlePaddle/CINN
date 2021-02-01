#pragma once

#include <mlir/IR/Function.h>
#include "cinnrt/host_context/core_runtime.h"
#include "cinnrt/host_context/function.h"
#include "cinnrt/host_context/mlir_to_runtime_translate.h"

namespace cinnrt {
namespace host_context {

struct KernelRegistry;

/**
 * Executable function for a given MLIR function definition, mainly used in two scenerios:
 * 1. cinn.call op
 * 2. main function call
 *
 * A MlirFunction might have one or more arguments and results.
 */
class MlirFunction : public Function, public MlirToRuntimeTranslator {
 public:
  /**
   * @param func_op a function IR node from the original MLIR module.
   * @param kernel_registry the kernel registry containing all the valid kernels.
   * @param function_table the symbol table for functions.
   */
  MlirFunction(mlir::FuncOp func_op,
               KernelRegistry* kernel_registry,
               MlirToRuntimeTranslator::function_table_t& function_table);

  /**
   * Execute the function with the given arguments and results.
   * NOTE the \param arguments and \param results should not be altered.
   */
  void Execute(llvm::ArrayRef<Value*> arguments, llvm::MutableArrayRef<ValueRef> results) const override;

 private:

  /**
   * Build the runtime executables once the function call arguments and results are passed in.
   * This will trigger in the first execution.
   */
  void BuildExecutables(llvm::ArrayRef<Value*> arguments, llvm::MutableArrayRef<ValueRef> results);

 private:
  mlir::FuncOp func_op_;
  CoreRuntimeBuilder core_runtime_;
  MlirToRuntimeTranslator::function_table_t& function_table_;
  std::function<void()> copy_res_fn_;
};

}  // namespace host_context
}  // namespace cinnrt

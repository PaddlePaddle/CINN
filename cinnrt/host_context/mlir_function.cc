#include "cinnrt/host_context/mlir_function.h"

namespace cinnrt {
namespace host_context {

MlirFunction::MlirFunction(mlir::FuncOp op)
    : Function(op.getName().str(), op.getNumArguments(), op.getNumResults()), func_op_(op) {}

void MlirFunction::Execute(llvm::ArrayRef<Value*> arguments, llvm::MutableArrayRef<ValueRef> results) const {
  CHECK_EQ(arguments.size(), num_arguments());
  CHECK_EQ(results.size(), num_results());

  auto& func_op = *const_cast<mlir::FuncOp*>(&func_op_);

  LOG(INFO) << "arg " << func_op.getArgument(0).getDefiningOp()->getName().getStringRef().str();
}

}  // namespace host_context
}  // namespace cinnrt

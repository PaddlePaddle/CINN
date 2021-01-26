#pragma once

#include <mlir/IR/Function.h>
#include "cinnrt/host_context/function.h"

namespace cinnrt {
namespace host_context {

class MlirFunction : public Function {
 public:
  explicit MlirFunction(mlir::FuncOp op);

  void Execute(llvm::ArrayRef<Value *> arguments, llvm::MutableArrayRef<ValueRef> results) const override;

 private:
  mlir::FuncOp func_op_;
};

}  // namespace host_context
}  // namespace cinnrt

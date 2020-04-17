#include "cinn/common/context.h"

#include "cinn/ir/ir.h"

namespace cinn {
namespace common {
using utils::any;

Context& Context::Global() {
  static Context x;
  return x;
}

const std::string& Context::runtime_llvm_ir_file() const {
  if (runtime_llvm_ir_file_.empty()) {
    runtime_llvm_ir_file_ = std::getenv(kRuntimeLlvmIrFileEnvironKey);
  }
  return runtime_llvm_ir_file_;
}

const char* kRuntimeLlvmIrFileEnvironKey = "runtime_llvm_ir_file";

}  // namespace common
}  // namespace cinn

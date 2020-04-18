#include "cinn/common/context.h"

#include "cinn/ir/ir.h"

namespace cinn {
namespace common {
using utils::any;

Context& Context::Global() {
  static Context x;
  return x;
}

const std::string& Context::runtime_include_dir() const {
  if (runtime_include_dir_.empty()) {
    runtime_include_dir_ = std::getenv(kRuntimeIncludeDirEnvironKey);
  }
  return runtime_include_dir_;
}

const char* kRuntimeLlvmIrFileEnvironKey = "runtime_llvm_ir_file";
const char* kRuntimeIncludeDirEnvironKey = "runtime_include_dir";

}  // namespace common
}  // namespace cinn

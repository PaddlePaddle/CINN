#include "cinn/common/context.h"

#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

Context& Context::Global() {
  static Context x;
  return x;
}

const std::string& Context::runtime_include_dir() const {
  if (runtime_include_dir_.empty()) {
    char* env            = std::getenv(kRuntimeIncludeDirEnvironKey);
    runtime_include_dir_ = env ? env : "";  // Leave empty if no env found.
  }
  return runtime_include_dir_;
}

const char* kRuntimeIncludeDirEnvironKey = "runtime_include_dir";

}  // namespace common

DEFINE_bool(cinn_runtime_display_debug_info, false, "Whether to display debug information in runtime");
}  // namespace cinn

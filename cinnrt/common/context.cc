#include "cinnrt/common/context.h"

namespace cinnrt {
namespace common {

Context& Context::Global() {
  static Context x;
  isl_options_set_on_error(x.ctx_.get(), ISL_ON_ERROR_ABORT);
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

std::string NameGenerator::New(const std::string& name_hint) {
  auto it = name_hint_idx_.find(name_hint);
  if (it == name_hint_idx_.end()) {
    name_hint_idx_.emplace(name_hint, -1);
    return name_hint;
  }
  return name_hint + "_" + std::to_string(++it->second);
}

}  // namespace common
}  // namespace cinnrt

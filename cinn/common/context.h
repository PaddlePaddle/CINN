#pragma once
#include <gflags/gflags.h>
#include <isl/cpp.h>

#include <any>
#include <set>
#include <string>
#include <vector>

#include "cinn/common/debug_manager.h"
#include "cinn/common/info_registry.h"
#include "cinn/common/target.h"

namespace cinn {

DECLARE_bool(cinn_runtime_display_debug_info);

namespace ir {
class Expr;
}  // namespace ir

namespace common {

extern const char* kRuntimeIncludeDirEnvironKey;

struct NameGenerator {
  std::string New(const std::string& name_hint);

  // Reset id to initial.
  void ResetID() { name_hint_idx_.clear(); }

 private:
  std::unordered_map<std::string, uint32_t> name_hint_idx_;
};

class Context {
 public:
  static Context& Global();

  /**
   * Generate a new unique name.
   * @param name_hint The prefix.
   */
  std::string NewName(const std::string& name_hint) { return name_generator_.New(name_hint); }
  void ResetNameId() { name_generator_.ResetID(); }

  InfoRegistry& info_rgt() { return info_rgt_; }

  DebugManager& debug_mgr() { return debug_mgr_; }

  const std::string& runtime_include_dir() const;

  /**
   * The global isl ctx.
   */
  isl::ctx isl_ctx() { return ctx_; }

 private:
  Context() : ctx_(isl_ctx_alloc()) {}
  NameGenerator name_generator_;
  isl::ctx ctx_;
  DebugManager debug_mgr_;
  InfoRegistry info_rgt_;

  mutable std::string runtime_include_dir_;
};

static std::string UniqName(const std::string& prefix) { return Context::Global().NewName(prefix); }

}  // namespace common
}  // namespace cinn

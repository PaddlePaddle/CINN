#pragma once
#include <isl/cpp.h>

#include <set>
#include <string>
#include <vector>

#include "cinn/common/debug_manager.h"
#include "cinn/common/info_registry.h"
#include "cinn/utils/any.h"

namespace cinn {

DECLARE_bool(cinn_runtime_display_debug_info);

namespace ir {
class Expr;
}  // namespace ir

namespace common {

struct ID {
  size_t id() const { return cur_; }
  size_t New() { return ++cur_; }
  void Reset() { cur_ = 0; }

 private:
  size_t cur_;
};

extern const char* kRuntimeIncludeDirEnvironKey;

struct NameGenerator {
  std::string New(const std::string& name_hint) { return name_hint + "_" + std::to_string(id_.New()); }

 private:
  ID id_;
};

class Context {
 public:
  static Context& Global();

  /**
   * Generate a new unique name.
   * @param name_hint The prefix.
   */
  std::string NewName(const std::string& name_hint) { return name_generator_.New(name_hint); }

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

}  // namespace common
}  // namespace cinn

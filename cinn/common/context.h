#pragma once
#include <isl/cpp.h>

#include <string>
#include <vector>

namespace cinn {
namespace common {

struct ID {
  size_t id() const { return cur_; }
  size_t New() { return ++cur_; }
  void Reset() { cur_ = 0; }

 private:
  size_t cur_;
};

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

  /**
   * The global isl ctx.
   */
  isl::ctx isl_ctx() { return ctx_; }

 private:
  Context() : ctx_(isl_ctx_alloc()) {}
  NameGenerator name_generator_;
  isl::ctx ctx_;
};

}  // namespace common
}  // namespace cinn

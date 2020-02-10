#pragma once
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
  static Context& Global() {
    static Context x;
    return x;
  }
  /**
   * Generate a new unique name.
   * @param name_hint The prefix.
   */
  std::string NewName(const std::string& name_hint) { return name_generator_.New(name_hint); }

 private:
  Context() = default;
  NameGenerator name_generator_;
};

}  // namespace common
}  // namespace cinn

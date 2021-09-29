#pragma once

#include <string>
#include <absl/strings/string_view.h>
#include <vector>

#include "cinn/lang/packed_func.h"

namespace cinn::ir {

class Registry {
 public:
  Registry &SetBody(lang::PackedFunc f);
  Registry &SetBody(lang::PackedFunc::body_t f);

  static Registry &Register(const std::string &name, bool can_override = false);
  static bool Remove(const std::string &name);
  static const lang::PackedFunc *Get(const std::string &name);
  static std::vector<std::string> ListNames();

  struct Manager;

  explicit Registry(const std::string &);

 protected:
  std::string name_;
  lang::PackedFunc func_;
  friend class Manager;
};

}  // namespace cinn::ir

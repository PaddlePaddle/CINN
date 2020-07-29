#pragma once

#include <string>
#include <string_view>
#include <vector>
#include "cinn/ir/packed_func.h"

namespace cinn::ir {

class Registry {
 public:
  Registry &SetBody(PackedFunc f);
  Registry &SetBody(PackedFunc::body_t f);

  static Registry &Register(const std::string &name, bool can_override = false);
  static bool Remove(const std::string &name);
  static const PackedFunc *Get(const std::string &name);
  static std::vector<std::string> ListNames();

  struct Manager;

  Registry(const std::string &);

 protected:
  std::string name_;
  PackedFunc func_;
  friend class Manager;
};

}  // namespace cinn::ir

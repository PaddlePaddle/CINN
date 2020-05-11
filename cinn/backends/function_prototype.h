#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace backends {

struct FunctionProto {
  FunctionProto(const std::string& name,
                const std::vector<Type>& readonly_arg_types,
                const std::vector<Type>& mutable_arg_types,
                Type ret_type = Void())
      : name(name), readonly_arg_types(readonly_arg_types), mutable_arg_types(mutable_arg_types), ret_type(ret_type) {}

  std::string name;
  std::vector<Type> readonly_arg_types;
  std::vector<Type> mutable_arg_types;
  Type ret_type;

  /**
   * Tell whether the Call \p op matches the function prototype.
   */
  bool Match(const ir::Call* op) const;

  /**
   * Assert the call should match the function prototype.
   */
  void AssertMatch(const ir::Call* op) const;
};

class FunctionProtoRegistry {
 public:
  FunctionProto* Register(std::string_view name, FunctionProto* x) {
    data_.emplace(name, std::unique_ptr<FunctionProto>(x));
    return x;
  }

  FunctionProto* Lookup(std::string_view name) {
    auto it = data_.find(name);
    if (it != data_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

 private:
  std::unordered_map<std::string_view, std::unique_ptr<FunctionProto>> data_;
};

}  // namespace backends
}  // namespace cinn

#include "cinnrt/host_context/symbol_table.h"

#include <string>

namespace cinnrt {
namespace host_context {

struct SymbolTable::Impl {
  std::unordered_map<std::string, ValueRef> data;
};

SymbolTable::SymbolTable() : impl_(new Impl) {}

Value* SymbolTable::Register(std::string_view key) {
  auto it = impl_->data.try_emplace(std::string(key), ValueRef(new Value));
  CHECK(it.second) << "Duplicate register [" << key << "]";
  return it.first->second.get();
}

Value* SymbolTable::Get(std::string_view key) const {
  auto it = impl_->data.find(std::string(key));
  return it != impl_->data.end() ? it->second.get() : nullptr;
}

SymbolTable::~SymbolTable() {}

#define REGISTER_TYPE__(T)                                            \
  template <>                                                         \
  Value* SymbolTable::Register(std::string_view key, T&& v) {         \
    auto it = impl_->data.try_emplace(std::string(key), ValueRef(v)); \
    CHECK(it.second) << "Duplicate register [" << key << "]";         \
    return it.first->second.get();                                    \
  }
REGISTER_TYPE__(int)
REGISTER_TYPE__(float)
REGISTER_TYPE__(double)
REGISTER_TYPE__(bool)
#undef REGISTER_TYPE__

}  // namespace host_context
}  // namespace cinnrt

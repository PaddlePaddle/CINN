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

Value* SymbolTable::Register(std::string_view key, ValueRef value) {
  auto it = impl_->data.try_emplace(std::string(key), value);
  CHECK(it.second) << "Duplicate register [" << key << "]";
  return it.first->second.get();
}

Value* SymbolTable::GetValue(std::string_view key) const {
  auto it = impl_->data.find(std::string(key));
  return it != impl_->data.end() ? it->second.get() : nullptr;
}

// @{
#define REGISTER_TYPE__(T)                                       \
  template <>                                                    \
  T SymbolTable::Get<T>(std::string_view key) {                  \
    auto it = impl_->data.find(std::string(key));                \
    CHECK(it != impl_->data.end()) << "No value called " << key; \
    return it->second->get<T>();                                 \
  }
REGISTER_TYPE__(int32_t);
REGISTER_TYPE__(float);
REGISTER_TYPE__(double);
REGISTER_TYPE__(int64_t);
#undef REGISTER_TYPE__
// @}

SymbolTable::~SymbolTable() {}

size_t SymbolTable::size() const { return impl_->data.size(); }

// @{
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
// @}

}  // namespace host_context
}  // namespace cinnrt

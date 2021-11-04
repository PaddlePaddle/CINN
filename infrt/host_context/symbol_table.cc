// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "infrt/host_context/symbol_table.h"

#include <string>

namespace infrt {
namespace host_context {

struct SymbolTable::Impl {
  absl::flat_hash_map<std::string, ValueRef> data;
};

SymbolTable::SymbolTable() : impl_(new Impl) {}

Value* SymbolTable::Register(absl::string_view key) {
  auto it = impl_->data.try_emplace(std::string(key), ValueRef(new Value));
  CHECK(it.second) << "Duplicate register [" << key << "]";
  return it.first->second.get();
}

Value* SymbolTable::Register(absl::string_view key, ValueRef value) {
  auto it = impl_->data.try_emplace(std::string(key), value);
  CHECK(it.second) << "Duplicate register [" << key << "]";
  return it.first->second.get();
}

Value* SymbolTable::GetValue(absl::string_view key) const {
  auto it = impl_->data.find(std::string(key));
  return it != impl_->data.end() ? it->second.get() : nullptr;
}

// @{
#define REGISTER_TYPE__(T)                                       \
  template <>                                                    \
  T SymbolTable::Get<T>(absl::string_view key) {                 \
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
  Value* SymbolTable::Register(absl::string_view key, T&& v) {        \
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
}  // namespace infrt

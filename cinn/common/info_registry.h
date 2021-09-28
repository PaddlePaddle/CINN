#pragma once
#include <string>
#include <absl/container/flat_hash_map.h>
#include <absl/types/any.h>

namespace cinn {
namespace common {

/**
 * Key value.
 */
class InfoRegistry {
 public:
  template <typename T>
  T& Get(const std::string& key);

  size_t size() const { return data_.size(); }

  void Clear() { data_.clear(); }

 private:
  absl::flat_hash_map<std::string, absl::any> data_;
};

template <typename T>
T& InfoRegistry::Get(const std::string& key) {
  auto it = data_.find(key);
  if (it == data_.end()) {
    data_[key] = T();
  }
  return absl::any_cast<T&>(data_[key]);
}

}  // namespace common
}  // namespace cinn

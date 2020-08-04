#pragma once
#include <any>
#include <string>
#include <unordered_map>

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
  std::unordered_map<std::string, std::any> data_;
};

template <typename T>
T& InfoRegistry::Get(const std::string& key) {
  auto it = data_.find(key);
  if (it == data_.end()) {
    data_[key] = T();
  }
  return std::any_cast<T&>(data_[key]);
}

}  // namespace common
}  // namespace cinn

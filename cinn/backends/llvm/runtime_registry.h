#pragma once

#include <map>
#include <mutex>  // NOLINT
#include <string>
#include <string_view>

namespace cinn::backends {

class RuntimeRegistry {
 public:
  static RuntimeRegistry *Global();

  template <typename T>
  void Register(std::string name, T *address) {
    std::lock_guard<std::mutex> lock(mu_);
    symbols_.insert({name, reinterpret_cast<void *>(address)});
  }

  void *Lookup(std::string_view name) const;
  const std::map<std::string, void *> &All() const { return symbols_; }

 private:
  RuntimeRegistry()                  = default;
  RuntimeRegistry(RuntimeRegistry &) = delete;
  void operator=(RuntimeRegistry &) = delete;

  mutable std::mutex mu_;
  std::map<std::string, void *> symbols_;
};
}  // namespace cinn::backends

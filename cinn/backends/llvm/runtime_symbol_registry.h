#pragma once

#include <glog/logging.h>

#include <absl/types/any.h>
#include <absl/types/variant.h>
#include <map>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include <absl/strings/string_view.h>

#include "cinn/common/macros.h"

namespace cinn {
namespace backends {

/**
 * Registry for runtime symbols, these symbols will be inserted into JIT.
 */
class RuntimeSymbolRegistry {
 public:
  static RuntimeSymbolRegistry &Global();

  /**
   * Register function address.
   * @param name Name of the symbol.
   * @param address Address of the function.
   */
  void RegisterFn(const std::string &name, void *address) { Register(name, address); }

  /**
   * Register scalar.
   * @tparam T Type of the scalar.
   * @param name Name of the symbol.
   * @param val Scalar value.
   */
  template <typename T>
  void RegisterVar(const std::string &name, T val) {
    auto &data = scalar_holder_[name];
    data.resize(sizeof(T));
    memcpy(data.data(), &val, sizeof(T));
    Register(name, reinterpret_cast<void *>(data.data()));
  }

  /**
   * Lookup a symbol from the registry.
   * @param name Name of the symbol.
   * @return The address if existes, or nullptr will return.
   */
  void *Lookup(absl::string_view name) const;

  /**
   * Get all the symbols.
   */
  const std::map<std::string, void *> &All() const { return symbols_; }

  /**
   * Clear all the symbols.
   */
  void Clear();

 private:
  /**
   * Register external symbol to the registry, the symbols in the registry will finally registered to JIT .
   * @param name Name of the symbol in the JIT.
   * @param address The address of the variable in external space.
   */
  void Register(const std::string &name, void *address);

  RuntimeSymbolRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(RuntimeSymbolRegistry);

  mutable std::mutex mu_;
  std::map<std::string, void *> symbols_;
  std::map<std::string, std::vector<int8_t>> scalar_holder_;
};

}  // namespace backends
}  // namespace cinn

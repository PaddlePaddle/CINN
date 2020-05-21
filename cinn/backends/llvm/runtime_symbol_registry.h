#pragma once

#include <glog/logging.h>

#include <map>
#include <mutex>  // NOLINT
#include <string>
#include <string_view>

#include "cinn/common/macros.h"

namespace cinn {
namespace backends {

class RuntimeSymbolRegistry {
 public:
  static RuntimeSymbolRegistry &Global();

  void Register(const std::string &name, void *address);

  void *Lookup(std::string_view name) const;
  const std::map<std::string, void *> &All() const { return symbols_; }

 private:
  RuntimeSymbolRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(RuntimeSymbolRegistry);

  mutable std::mutex mu_;
  std::map<std::string, void *> symbols_;
};

}  // namespace backends
}  // namespace cinn

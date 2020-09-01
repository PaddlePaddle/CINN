#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace hlir {
namespace framework {

Variable* Scope::FindVar(const std::string& name) const {
  auto it = dic.find(name);
  if (it != dic.end()) return it->second.get();
  return nullptr;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace hlir {
namespace framework {

Variable* Scope::FindVar(const std::string& name) const {
  CHECK(utils::IsVarNameValid(name));
  auto it = data_.find(name);
  if (it != data_.end()) return it->second.get();
  return nullptr;
}

Tensor* Scope::GetTensor(const std::string& name) const {
  CHECK(utils::IsVarNameValid(name));
  auto* var = FindVar(name);
  CHECK(var) << "No variable called [" << name << "] found";
  return &std::get<Tensor>(*var);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

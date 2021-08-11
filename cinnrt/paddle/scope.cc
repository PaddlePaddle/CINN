#include "cinnrt/paddle/scope.h"

#include "cinnrt/common/common.h"

namespace cinnrt {
namespace paddle {

_Variable* Scope::FindVar(const std::string& name) const {
  auto it = data_.find(name);
  if (it != data_.end()) return it->second.get();
  return nullptr;
}

Tensor Scope::GetTensor(const std::string& name) const {
  CheckVarNameValid(name);
  auto* var = FindVar(name);
  CHECK(var) << "No variable called [" << name << "] found";
  return std::get<Tensor>(*var);
}

std::vector<std::string_view> Scope::var_names() const {
  std::vector<std::string_view> names;
  for (auto& item : data_) {
    names.push_back(item.first);
  }
  return names;
}

}  // namespace paddle
}  // namespace cinnrt

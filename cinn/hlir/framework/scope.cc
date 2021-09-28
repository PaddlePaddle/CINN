#include "cinn/hlir/framework/scope.h"

#include "cinn/common/common.h"

namespace cinn {
namespace hlir {
namespace framework {

Variable* Scope::FindVar(const std::string& name) const {
  auto it = data_.find(name);
  if (it != data_.end()) return it->second.get();
  return nullptr;
}

Tensor Scope::GetTensor(const std::string& name) const {
  CheckVarNameValid(name);
  auto* var = FindVar(name);
  CHECK(var) << "No variable called [" << name << "] found";
  return absl::get<Tensor>(*var);
}

std::vector<absl::string_view> Scope::var_names() const {
  std::vector<absl::string_view> names;
  for (auto& item : data_) {
    names.push_back(item.first);
  }
  return names;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

#pragma once
#include <absl/container/flat_hash_map.h>
#include <memory>
#include <string>
#include <vector>

#include <absl/strings/string_view.h>
#include <absl/types/any.h>
#include <absl/types/variant.h>

#include "cinn/common/macros.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

using Variable = absl::variant<Tensor>;

struct _Tensor_;

class Scope {
 public:
  static std::shared_ptr<Scope> Create() { return std::make_shared<Scope>(); }

  //! Get or create a variable.
  template <typename T>
  Variable* Var(const std::string& name);

  //! Find a variable, get null if not exists.
  Variable* FindVar(const std::string& name) const;

  Tensor GetTensor(const std::string& name) const;

  //! Get variable names.
  std::vector<absl::string_view> var_names() const;

  Scope() = default;

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<Variable>> data_;

  CINN_DISALLOW_COPY_AND_ASSIGN(Scope);
};

template <typename T>
Variable* Scope::Var(const std::string& name) {
  VLOG(4) << "Scope insert Var [" << name << "]";
  Variable* x = FindVar(name);
  if (x) return x;
  auto* data = new Variable(T());
  data_[name].reset(data);
  return data;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

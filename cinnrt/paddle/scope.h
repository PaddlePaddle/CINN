#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/types/variant.h>
#include <memory>
#include <string>
#include <vector>

#include "cinnrt/common/macros.h"
#include "cinnrt/paddle/tensor.h"

namespace cinnrt {
namespace paddle {

using _Variable = absl::variant<Tensor>;

struct _Tensor_;

class Scope {
 public:
  static std::shared_ptr<Scope> Create() { return std::make_shared<Scope>(); }

  //! Get or create a variable.
  template <typename T>
  _Variable* Var(const std::string& name);

  //! Find a variable, get null if not exists.
  _Variable* FindVar(const std::string& name) const;

  Tensor GetTensor(const std::string& name) const;

  //! Get variable names.
  std::vector<absl::string_view> var_names() const;

  Scope() = default;

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<_Variable>> data_;

  CINN_DISALLOW_COPY_AND_ASSIGN(Scope);
};

template <typename T>
_Variable* Scope::Var(const std::string& name) {
  VLOG(4) << "Scope insert Var [" << name << "]";
  _Variable* x = FindVar(name);
  if (x) return x;
  auto* data = new _Variable(T());
  data_[name].reset(data);
  return data;
}

}  // namespace paddle
}  // namespace cinnrt

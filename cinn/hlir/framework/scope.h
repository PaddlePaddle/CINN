#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "cinn/common/macros.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

using Variable = std::variant<Tensor>;

struct Tensor;

class Scope {
 public:
  static std::shared_ptr<Scope> Create() { return std::make_shared<Scope>(); }

  //! Get or create a variable.
  template <typename T>
  Variable* Var(const std::string& name);

  //! Find a variable, get null if not exists.
  Variable* FindVar(const std::string& name) const;

  Tensor* GetTensor(const std::string& name) const;

  //! Get variable names.
  std::vector<std::string_view> var_names() const;

  Scope() = default;

 private:
  std::unordered_map<std::string, std::unique_ptr<Variable>> data_;

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

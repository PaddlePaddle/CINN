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

class Scope {
 public:
  Scope() = default;

  //! Get or create a variable.
  template <typename T>
  Variable* Var(const std::string& name);

  //! Find a variable, get null if not exists.
  Variable* FindVar(const std::string& name);

 private:
  std::unordered_map<std::string, std::unique_ptr<Variable>> dic;

  CINN_DISALLOW_COPY_AND_ASSIGN(Scope);
};

template <typename T>
Variable* Scope::Var(const std::string& name) {
  Variable* x = FindVar(name);
  if (x) return x;
  auto* data = new Variable(T());
  dic[name].reset(data);
  return data;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

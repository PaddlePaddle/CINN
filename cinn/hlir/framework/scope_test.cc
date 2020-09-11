#include "cinn/hlir/framework/scope.h"

#include <gtest/gtest.h>

namespace cinn {
namespace hlir {
namespace framework {

TEST(Scope, basic) {
  Scope scope;
  auto* var    = scope.Var<Tensor>("key");
  auto& tensor = std::get<Tensor>(*var);
  tensor->Resize(Shape{{3, 1}});
  auto* data = tensor->mutable_data<float>(common::DefaultHostTarget());
  data[0]    = 0.f;
  data[1]    = 1.f;
  data[2]    = 2.f;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

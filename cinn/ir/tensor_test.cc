#include "cinn/ir/tensor.h"
#include <gtest/gtest.h>

namespace cinn {
namespace ir {

TEST(Tensor, test) {
  Var M, N;
  Tensor tensor({M, N});
}

}  // namespace ir
}  // namespace cinn
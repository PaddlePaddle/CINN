#include "cinn/hlir/framework/tensor.h"

#include <gtest/gtest.h>

namespace cinn {
namespace hlir {
namespace framework {

TEST(Tensor, basic) {
  Tensor tensor;
  tensor.Resize(Shape{{3, 2}});

  auto* data = tensor.mutable_data<float>(common::DefaultHostTarget());

  for (int i = 0; i < tensor.shape().numel(); i++) {
    data[i] = i;
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

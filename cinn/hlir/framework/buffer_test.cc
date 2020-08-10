#include "cinn/hlir/framework/buffer.h"

#include <gtest/gtest.h>

namespace cinn {
namespace hlir {
namespace framework {

TEST(Buffer, basic) {
  Buffer buffer(common::DefaultHostTarget());
  buffer.Resize(10 * sizeof(float));
  auto* data = reinterpret_cast<float*>(buffer.data());
  for (int i = 0; i < 10; i++) data[i] = i;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

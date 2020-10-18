#include "cinn/host_context/kernel_utils.h"
#include <gtest/gtest.h>

namespace cinn::host_context {

int add(int a, int b) { return a + b; }
TEST(KernelImpl, basic) {
  KernelFrameBuilder fbuilder;
  fbuilder.AddArgument(Value(1));
  fbuilder.AddArgument(Value(2));
  fbuilder.SetNumResults(1);

  CINN_KERNEL(add)(&fbuilder);
  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results.front().get<int>(), 3);
}

}  // namespace cinn::host_context

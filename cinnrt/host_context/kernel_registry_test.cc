#include "cinnrt/host_context/kernel_registry.h"

#include <gtest/gtest.h>

#include "cinnrt/host_context/kernel_utils.h"

namespace cinnrt::host_context {

int add_i32(int a, int b) { return a + b; }

TEST(KernelRegistry, basic) {
  KernelRegistry registry;
  std::string_view key = "cinn.test.add.i32";
  registry.AddKernel(key, CINN_KERNEL(add_i32));

  auto* kernel_impl = registry.GetKernel(key);
  ASSERT_TRUE(kernel_impl);

  KernelFrameBuilder fbuilder;
  fbuilder.AddArgument(ValueRef(1));
  fbuilder.AddArgument(ValueRef(2));
  fbuilder.SetNumResults(1);

  kernel_impl(&fbuilder);

  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 1UL);
  ASSERT_EQ(results[0].get<int>(), 3);
}

}  // namespace cinnrt::host_context

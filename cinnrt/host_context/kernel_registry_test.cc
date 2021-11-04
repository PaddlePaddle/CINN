#include "infrt/host_context/kernel_registry.h"

#include <gtest/gtest.h>

#include "infrt/host_context/kernel_utils.h"

namespace infrt::host_context {

int add_i32(int a, int b) { return a + b; }

TEST(KernelRegistry, basic) {
  KernelRegistry registry;
  std::string key = "cinn.test.add.i32";
  registry.AddKernel(key, CINN_KERNEL(add_i32));

  auto* kernel_impl = registry.GetKernel(key);
  ASSERT_TRUE(kernel_impl);

  ValueRef a(1);
  ValueRef b(2);
  KernelFrameBuilder fbuilder;
  fbuilder.AddArgument(a.get());
  fbuilder.AddArgument(b.get());
  fbuilder.SetNumResults(1);

  kernel_impl(&fbuilder);

  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 1UL);
  ASSERT_EQ(results[0]->get<int>(), 3);
}

}  // namespace infrt::host_context

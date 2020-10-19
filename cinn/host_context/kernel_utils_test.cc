#include "cinn/host_context/kernel_utils.h"
#include <gtest/gtest.h>

namespace cinn::host_context {

int add_i32(int a, int b) { return a + b; }
float add_f32(float a, float b) { return a + b; }
std::pair<int, float> add_pair(int a, float b) { return {a, b}; }

TEST(KernelImpl, i32) {
  KernelFrameBuilder fbuilder;
  fbuilder.AddArgument(ValueRef(1));
  fbuilder.AddArgument(ValueRef(2));
  fbuilder.SetNumResults(1);

  CINN_KERNEL(add_i32)(&fbuilder);
  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results.front().get<int>(), 3);
}

TEST(KernelImpl, f32) {
  KernelFrameBuilder fbuilder;
  fbuilder.AddArgument(ValueRef(1.f));
  fbuilder.AddArgument(ValueRef(2.f));
  fbuilder.SetNumResults(1);

  CINN_KERNEL(add_f32)(&fbuilder);
  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results.front().get<float>(), 3.f);
}

TEST(KernelImpl, pair) {
  KernelFrameBuilder fbuilder;
  fbuilder.AddArgument(ValueRef(1));
  fbuilder.AddArgument(ValueRef(3.f));
  fbuilder.SetNumResults(2);

  CINN_KERNEL(add_pair)(&fbuilder);
  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 2UL);
  ASSERT_EQ(results[0].get<int>(), 1);
  ASSERT_EQ(results[1].get<float>(), 3.f);
}

}  // namespace cinn::host_context

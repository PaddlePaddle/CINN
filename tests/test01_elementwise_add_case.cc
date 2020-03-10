#include <gtest/gtest.h>

#include "cinn/runtime/cinn_runtime.h"
#include "tests/test01_elementwise_add.h"

TEST(test01, basic) {
  auto* A = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {100, 20});
  auto* B = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {100, 20});
  auto* C = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {100, 20});
  cinn_buffer_malloc(nullptr, A);
  cinn_buffer_malloc(nullptr, B);

  float* Ad = reinterpret_cast<float*>(A->host_memory);
  float* Bd = reinterpret_cast<float*>(B->host_memory);

  for (int i = 0; i < A->num_elements(); i++) {
    Ad[i] = i;
    Bd[i] = i;
  }

  add1(A, B, C);

  float* Cd = reinterpret_cast<float*>(C->host_memory);
  ASSERT_EQ(C->num_elements(), A->num_elements());

  for (int i = 0; i < C->num_elements(); i++) {
    EXPECT_EQ(Ad[i] + Bd[i], Cd[i]);
  }
}

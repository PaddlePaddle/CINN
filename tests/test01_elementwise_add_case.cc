#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/common/test_helper.h"
#include "cinn/runtime/cinn_runtime.h"
#include "tests/test01_elementwise_add.h"
#include "tests/test01_elementwise_add_compute_at.h"
#include "tests/test01_elementwise_add_compute_at_level1.h"
#include "tests/test01_elementwise_add_vectorize.h"

namespace cinn {

TEST(test01, basic) {
  auto* A = cinn::common::BufferBuilder(Float(32), {100, 32}).set_align(32).set_random().Build();
  auto* B = cinn::common::BufferBuilder(Float(32), {100, 32}).set_align(32).set_random().Build();
  auto* C = cinn::common::BufferBuilder(Float(32), {100, 32}).set_align(32).set_zero().Build();

  float* Ad = reinterpret_cast<float*>(A->memory);
  float* Bd = reinterpret_cast<float*>(B->memory);
  float* Cd = reinterpret_cast<float*>(C->memory);
  ASSERT_EQ(C->num_elements(), A->num_elements());

  auto check = [&] {
    for (int i = 0; i < C->num_elements(); i++) {
      EXPECT_EQ(Ad[i] + Bd[i], Cd[i]);
    }
  };

  auto args = common::ArgsBuilder().Add(A).Add(B).Add(C).Build();

  LOG(INFO) << "test1 basic";
  add1(args.data(), args.size());
  check();

  LOG(INFO) << "test1 vectorize";
  add1_vectorize(args.data(), args.size());
  check();
}

TEST(test01, compute_at) {
  const int M = 100;
  const int N = 32;
  auto* A     = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_random().Build();
  auto* B     = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_random().Build();
  auto* C     = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_zero().Build();

  float* Ad = reinterpret_cast<float*>(A->memory);
  float* Bd = reinterpret_cast<float*>(B->memory);
  float* Cd = reinterpret_cast<float*>(C->memory);
  ASSERT_EQ(C->num_elements(), A->num_elements());

  auto check_add = [&] {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_NEAR(Ad[i * N + j] + Bd[i * N + j], Cd[i * N + j], 1e-5);
      }
    }
  };

  auto check_compute = [&] {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float first = i > 0 ? Ad[(i - 1) * N + j] : 0.f;
        float last  = i < M - 1 ? Ad[(i + 1) * N + j] : 0.f;
        float left  = first + last + Ad[i * N + j] + Bd[i * N + j];
        ASSERT_NEAR(left, Cd[i * N + j], 1e-5);
      }
    }
  };

  auto args = common::ArgsBuilder().Add(A).Add(B).Add(C).Build();

  LOG(INFO) << "test1 basic";
  add1(args.data(), args.size());
  check_add();

  LOG(INFO) << "test1 vectorize";
  fn_compute_at(args.data(), args.size());
  check_compute();

  cinn_buffer_free(nullptr, A);
  cinn_buffer_free(nullptr, B);
  cinn_buffer_free(nullptr, C);
}

TEST(test01, compute_at_level1) {
  const int M = 100;
  const int N = 32;
  auto* A     = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_random().Build();
  auto* B     = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_random().Build();
  auto* C     = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_zero().Build();

  float* Ad = reinterpret_cast<float*>(A->memory);
  float* Bd = reinterpret_cast<float*>(B->memory);
  float* Cd = reinterpret_cast<float*>(C->memory);
  ASSERT_EQ(C->num_elements(), A->num_elements());

  auto check_add = [&] {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_NEAR(Ad[i * N + j] + Bd[i * N + j], Cd[i * N + j], 1e-5);
      }
    }
  };

  auto check_compute = [&] {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float first = i > 0 ? Ad[(i - 1) * N + j] : 0.f;
        float last  = i < M - 1 ? Ad[(i + 1) * N + j] : 0.f;
        float left  = first + last + Ad[i * N + j] + Bd[i * N + j];
        ASSERT_NEAR(left, Cd[i * N + j], 1e-5);
      }
    }
  };

  auto args = common::ArgsBuilder().Add(A).Add(B).Add(C).Build();

  LOG(INFO) << "test1 basic";
  add1(args.data(), args.size());
  check_add();

  LOG(INFO) << "test1 vectorize";
  fn_compute_at_level1(args.data(), args.size());
  check_compute();

  cinn_buffer_free(nullptr, A);
  cinn_buffer_free(nullptr, B);
  cinn_buffer_free(nullptr, C);
}

}  // namespace cinn

// include the generated C source code:
// @{
#include "tests/test01_elementwise_add.cc"
#include "tests/test01_elementwise_add_compute_at.cc"
#include "tests/test01_elementwise_add_compute_at_level1.cc"
#include "tests/test01_elementwise_add_vectorize.cc"
// @}

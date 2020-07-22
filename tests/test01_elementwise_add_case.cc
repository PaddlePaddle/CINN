#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/common/test_helper.h"
#include "cinn/runtime/cinn_runtime.h"
#include "tests/test01_elementwise_add.h"
#include "tests/test01_elementwise_add_compute_at.h"
#include "tests/test01_elementwise_add_vectorize.h"

TEST(test01, basic) {
  using namespace cinn;
  auto* A = cinn::common::BufferBuilder(Float(32), {100, 32}).set_align(32).set_random().Build();
  auto* B = cinn::common::BufferBuilder(Float(32), {100, 32}).set_align(32).set_random().Build();
  auto* C = cinn::common::BufferBuilder(Float(32), {100, 32}).set_align(32).set_zero().Build();

  float* Ad = reinterpret_cast<float*>(A->host_memory);
  float* Bd = reinterpret_cast<float*>(B->host_memory);
  float* Cd = reinterpret_cast<float*>(C->host_memory);
  ASSERT_EQ(C->num_elements(), A->num_elements());

  auto check = [&] {
    for (int i = 0; i < C->num_elements(); i++) {
      EXPECT_EQ(Ad[i] + Bd[i], Cd[i]);
    }
  };

  cinn_pod_value_t A_arg(A);
  cinn_pod_value_t B_arg(B);
  cinn_pod_value_t C_arg(C);
  cinn_pod_value_t args[] = {A_arg, B_arg, C_arg};

  LOG(INFO) << "test1 basic";
  add1(args, 3);
  check();

  LOG(INFO) << "test1 vectorize";
  add1_vectorize(args, 3);
  check();
}

TEST(test01, compute_at) {
  const int M = 100;
  const int N = 32;
  using namespace cinn;
  auto* A = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_random().Build();
  auto* B = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_random().Build();
  auto* C = cinn::common::BufferBuilder(Float(32), {M, N}).set_align(32).set_zero().Build();

  float* Ad = reinterpret_cast<float*>(A->host_memory);
  float* Bd = reinterpret_cast<float*>(B->host_memory);
  float* Cd = reinterpret_cast<float*>(C->host_memory);
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

  cinn_pod_value_t A_arg(A);
  cinn_pod_value_t B_arg(B);
  cinn_pod_value_t C_arg(C);
  cinn_pod_value_t args[] = {A_arg, B_arg, C_arg};

  LOG(INFO) << "test1 basic";
  add1(args, 3);
  check_add();

  LOG(INFO) << "test1 vectorize";
  fn_compute_at(args, 3);
  check_compute();
}

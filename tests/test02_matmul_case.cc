#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/runtime/cinn_runtime.h"
#include "cinn/utils/timer.h"
#include "tests/test02_helper.h"
#include "tests/test02_matmul.h"
#include "tests/test02_matmul_array_packing.h"
#include "tests/test02_matmul_array_packing_dynamic_shape.h"
#include "tests/test02_matmul_block.h"
#include "tests/test02_matmul_call.h"
#include "tests/test02_matmul_loop_permutation.h"
#include "tests/test02_matmul_split.h"
#include "tests/test02_matmul_tile.h"
#include "tests/test02_matmul_varient_shape.h"
#include "tests/test02_matmul_varient_shape_tile.h"
#include "tests/test02_matmul_vectorize.h"

TEST(test02, basic) {
  const int M  = 1024;
  const int N  = 1024;
  const int K  = 1024;
  const int bn = 32;

  auto* A        = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, K}, 32);
  auto* B        = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {K, N}, 32);
  auto* C        = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, N}, 32);
  auto* C_target = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, N});
  auto* packedB  = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {N / bn, K, bn}, 32);
  cinn_buffer_malloc(nullptr, A);
  cinn_buffer_malloc(nullptr, B);
  cinn_buffer_malloc(nullptr, C_target);
  cinn_buffer_malloc(nullptr, C);
  cinn_buffer_malloc(nullptr, packedB);

  float* Ad        = reinterpret_cast<float*>(A->host_memory);
  float* Bd        = reinterpret_cast<float*>(B->host_memory);
  float* Cd_target = reinterpret_cast<float*>(C_target->host_memory);
  float* Cd        = reinterpret_cast<float*>(C->host_memory);

  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      Ad[i * K + k] = float(rand()) / RAND_MAX;  // NOLINT
    }
  }

  for (int j = 0; j < M; j++) {
    for (int k = 0; k < K; k++) {
      Bd[k * N + j] = float(rand()) / RAND_MAX;  // NOLINT
    }
  }

  // manually set zero
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      Cd_target[i * N + j] = 0.f;
      // Cd[i * N + j]        = 0.f;
    }
  }

  auto compare = [&](float diff = 1e-5) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_NEAR(Cd[i * N + j], Cd_target[i * N + j], diff);
      }
    }
  };

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        Cd_target[i * N + j] += Ad[i * K + k] * Bd[k * N + j];
      }
    }
  }

  cinn::utils::Timer timer;

  const int repeat = 2;

  cinn_pod_value_t A_arg(A);
  cinn_pod_value_t B_arg(B);
  cinn_pod_value_t C_arg(C);
  cinn_pod_value_t packedB_arg(packedB);
  cinn_pod_value_t M_arg(M);

  cinn_pod_value_t args[]  = {A_arg, B_arg, C_arg};
  cinn_pod_value_t args1[] = {A_arg, B_arg, C_arg, packedB_arg};
  cinn_pod_value_t args2[] = {M_arg, A_arg, B_arg, C_arg};
  cinn_pod_value_t args3[] = {M_arg, A_arg, B_arg, C_arg};

#define TEST_FUNC(func__)                                                     \
  LOG(INFO) << "Testing " #func__;                                            \
  timer.Start();                                                              \
  for (int i = 0; i < repeat; i++) func__(reinterpret_cast<void**>(args), 3); \
  LOG(INFO) << timer.Stop() / repeat;                                         \
  compare();

#define TEST_FUNC1(func__, diff)                                               \
  LOG(INFO) << "Testing " #func__;                                             \
  timer.Start();                                                               \
  for (int i = 0; i < repeat; i++) func__(reinterpret_cast<void**>(args1), 4); \
  LOG(INFO) << timer.Stop() / repeat;                                          \
  compare();

#define TEST_FUNC2(func__, diff)                                               \
  LOG(INFO) << "Testing " #func__;                                             \
  timer.Start();                                                               \
  for (int i = 0; i < repeat; i++) func__(reinterpret_cast<void**>(args2), 4); \
  LOG(INFO) << timer.Stop() / repeat;                                          \
  compare();

#define TEST_FUNC3(func__, diff)                                               \
  LOG(INFO) << "Testing " #func__;                                             \
  timer.Start();                                                               \
  for (int i = 0; i < repeat; i++) func__(reinterpret_cast<void**>(args3), 4); \
  LOG(INFO) << timer.Stop() / repeat;                                          \
  compare();

  TEST_FUNC(matmul)

  TEST_FUNC(matmul_tile)

  TEST_FUNC(matmul_split)

  TEST_FUNC(matmul_block)

  TEST_FUNC(matmul_vectorize)

  TEST_FUNC(matmul_loop_permutation)

  TEST_FUNC1(matmul_array_packing, 1e-5)

  TEST_FUNC2(matmul_dynamic_shape, 1e-5);

  TEST_FUNC2(matmul_dynamic_shape_tile, 1e-5);

  TEST_FUNC3(matmul_array_packing_dynamic_shape, 1e-5);

  TEST_FUNC(matmul_main);

  {
    auto module    = cinn::tests::CreateModule("module", 1024, 1024, 1024);
    auto jit       = cinn::tests::CreateSimpleOrcJit(module);
    auto matmul_fn = reinterpret_cast<void (*)(void**, int32_t)>(jit->Lookup("matmul"));
    TEST_FUNC(matmul_fn);
  }

  {
    auto module    = cinn::tests::CreateModule("module", 1024, 1024, 1024);
    auto jit       = cinn::tests::CreateSimpleJit(module);
    auto matmul_fn = reinterpret_cast<void (*)(void**, int32_t)>(jit->Lookup("matmul"));
    TEST_FUNC(matmul_fn);
  }
}

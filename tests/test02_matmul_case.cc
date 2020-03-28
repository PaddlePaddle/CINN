#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/runtime/cinn_runtime.h"
#include "cinn/utils/timer.h"
#include "tests/test02_matmul.h"
#include "tests/test02_matmul_array_packing.h"
#include "tests/test02_matmul_block.h"
#include "tests/test02_matmul_loop_permutation.h"
#include "tests/test02_matmul_split.h"
#include "tests/test02_matmul_tile.h"
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
        EXPECT_NEAR(Cd[i * N + j], Cd_target[i * N + j], diff);
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

  const int repeat = 1;

#define TEST_FUNC(func__)                           \
  LOG(INFO) << "Testing " #func__;                  \
  timer.Start();                                    \
  for (int i = 0; i < repeat; i++) func__(A, B, C); \
  LOG(INFO) << timer.Stop() / repeat;               \
  compare();

#define TEST_FUNC1(func__, diff)                             \
  LOG(INFO) << "Testing " #func__;                           \
  timer.Start();                                             \
  for (int i = 0; i < repeat; i++) func__(A, B, C, packedB); \
  LOG(INFO) << timer.Stop() / repeat;                        \
  compare();

  TEST_FUNC(matmul)

  TEST_FUNC(matmul_tile)

  TEST_FUNC(matmul_split)

  TEST_FUNC(matmul_block)

  TEST_FUNC(matmul_vectorize)

  TEST_FUNC(matmul_loop_permutation)

  TEST_FUNC1(matmul_array_packing, 1e-5)
}

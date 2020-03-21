#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/runtime/cinn_runtime.h"
#include "cinn/utils/timer.h"
#include "tests/test02_matmul.h"
#include "tests/test02_matmul_block.h"
#include "tests/test02_matmul_split.h"
#include "tests/test02_matmul_tile.h"
#include "tests/test02_matmul_vectorize.h"

TEST(test02, basic) {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  auto* A        = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, K}, 32);
  auto* B        = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {K, N}, 32);
  auto* C        = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, N}, 32);
  auto* C_target = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {M, N});
  cinn_buffer_malloc(nullptr, A);
  cinn_buffer_malloc(nullptr, B);
  cinn_buffer_malloc(nullptr, C_target);
  cinn_buffer_malloc(nullptr, C);

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

  auto compare = [&]() {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        EXPECT_NEAR(Cd[i * N + j], Cd_target[i * N + j], 1e-5);
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

  LOG(INFO) << "Testing matmul_basic";
  timer.Start();
  for (int i = 0; i < repeat; i++) matmul(A, B, C);
  LOG(INFO) << timer.Stop() / repeat;
  compare();

  LOG(INFO) << "Testing matmul_tile";
  timer.Start();
  for (int i = 0; i < repeat; i++) matmul_tile(A, B, C);
  LOG(INFO) << timer.Stop() / repeat;
  compare();

  LOG(INFO) << "Testing matmul_split";
  timer.Start();
  for (int i = 0; i < repeat; i++) matmul_split(A, B, C);
  LOG(INFO) << timer.Stop() / repeat;
  compare();

  LOG(INFO) << "Testing matmul_block";
  timer.Start();
  for (int i = 0; i < repeat; i++) matmul_block(A, B, C);
  LOG(INFO) << timer.Stop() / repeat;
  compare();

  LOG(INFO) << "Testing matmul_vectorize";
  timer.Start();
  for (int i = 0; i < repeat; i++) matmul_vectorize(A, B, C);
  LOG(INFO) << timer.Stop() / repeat;
  compare();
}

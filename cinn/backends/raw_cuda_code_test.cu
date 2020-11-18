#include <glog/logging.h>
#include <gtest/gtest.h>
#include "cinn/backends/cuda_util.h"
#include "cinn/utils/timer.h"

__global__ void elementwise_add_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C) {
  if ((blockIdx.x < 1024)) {
    {
      if ((threadIdx.x < 1024)) {
        {
          C[((1024 * blockIdx.x) + threadIdx.x)] =
              (A[((1024 * blockIdx.x) + threadIdx.x)] + B[((1024 * blockIdx.x) + threadIdx.x)]);
        }
      }
    }
  }
}

TEST(raw_cuda, basic) {
  const int M = 1024;
  const int N = 1024;
  // allocate CUDA buffer
  float *Ag, *Bg, *Cg;
  const int num_bytes = M * N * sizeof(float);
  cudaMalloc(&Ag, num_bytes);
  cudaMalloc(&Bg, num_bytes);
  cudaMalloc(&Cg, num_bytes);

  cinn::utils::Timer timer;
  timer.Start();
  for (int i = 0; i < 1000; i++) {
    elementwise_add_kernel<<<1024, 1024>>>(Ag, Bg, Cg);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  float latency = timer.Stop();
  LOG(INFO) << "latency: " << latency / 1000;
}

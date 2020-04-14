#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL) << "CUDAError: " #x " failed with error: " << msg;     \
    }                                                                   \
  }

#define CUDA_CALL(func)                                                                            \
  {                                                                                                \
    cudaError_t e = (func);                                                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) << "CUDA: " << cudaGetErrorString(e); \
  }

#define NVRTC_CALL(x)                                                                         \
  {                                                                                           \
    nvrtcResult result = x;                                                                   \
    if (result != NVRTC_SUCCESS) {                                                            \
      LOG(FATAL) << "NVRTC error: " #x " failed with error: " << nvrtcGetErrorString(result); \
    }                                                                                         \
  }

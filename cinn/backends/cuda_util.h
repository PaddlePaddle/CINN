#pragma once
#ifdef CINN_WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <string>
#include <tuple>
#include <vector>

#include "cinn/runtime/cinn_runtime.h"

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

#define CUDNN_CALL(f)                                                                                       \
  {                                                                                                         \
    cudnnStatus_t err = (f);                                                                                \
    CHECK(err == CUDNN_STATUS_SUCCESS) << "    Error occurred: " << cudnnGetErrorString(err) << " on line " \
                                       << __LINE__ << std::endl;                                            \
  }

#define NVRTC_CALL(x)                                                                         \
  {                                                                                           \
    nvrtcResult result = x;                                                                   \
    if (result != NVRTC_SUCCESS) {                                                            \
      LOG(FATAL) << "NVRTC error: " #x " failed with error: " << nvrtcGetErrorString(result); \
    }                                                                                         \
  }

namespace cinn {
namespace backends {

// CUDA syntax for thread axis.
std::string cuda_thread_axis_name(int level);

// CUDA syntax for block axis.
std::string cuda_block_axis_name(int level);

}  // namespace backends
}  // namespace cinn

#endif  // CINN_WITH_CUDA

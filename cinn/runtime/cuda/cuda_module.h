#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/runtime/cuda/cuda_util.h"

namespace cinn {
namespace runtime {
namespace cuda {

/**
 * The CUDA module, helps to compile CUDA codes and fetch symbols.
 * Currently, it is a wrapper of NVRTC.
 */
class CUDAModule {
 public:
  enum class Kind {
    PTX = 0,
  };

  CUDAModule(const std::string& data, Kind kind) : data_(data), kind_(kind) {
    CHECK(!data.empty());

    cudaGetDeviceCount(&num_devices_);

    // TODO(Superjomn) Determine whether to initialize all the devices.
    cuInit(0);
    cuDeviceGet(&device_, 0);
    cuCtxCreate(&context_, 0, device_);
  }

  void LaunchKernel(int device_id,
                    const std::string& func_name,
                    dim3 gridDim,
                    dim3 blockDim,
                    void** args,
                    size_t share_memory_size = 0,
                    CUstream stream          = nullptr);

  //! Get a function.
  CUfunction GetFunction(int device_id, const std::string& func_name);

  //! Get a global variable.
  CUdeviceptr GetGlobal(int device_id, const std::string& name, size_t nbytes) {
    if (!module_per_card_[device_id]) {
      std::lock_guard<std::mutex> lock(mutex_);
      CUDA_DRIVER_CALL(cuModuleLoadData(&module_per_card_[device_id], data_.c_str()));
    }

    CUdeviceptr global;
    size_t _nbytes;
    CUDA_DRIVER_CALL(cuModuleGetGlobal(&global, &_nbytes, module_per_card_[device_id], name.c_str()));
    return global;
  }

  ~CUDAModule() {
    for (int i = 0; i < module_per_card_.size(); i++) {
      auto* module = module_per_card_[i];
      if (module) {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_DRIVER_CALL(cuModuleUnload(module));
      }
    }
  }

 private:
  //! The input data.
  std::string data_;
  //! Kind of the input.
  Kind kind_;
  //! To make parallel, we prepare one module for each card.
  std::vector<CUmodule> module_per_card_{kCUDAMaxCards, nullptr};
  std::string cuda_source_;
  std::mutex mutex_;

  CUdevice device_;
  CUcontext context_;
  int num_devices_{0};
};

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn

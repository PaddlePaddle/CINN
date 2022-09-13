// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/runtime/cuda/cuda_module.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <mutex>  // NOLINT
#include <string>

#include "cinn/backends/cuda_util.h"
#include "cinn/runtime/cuda/cuda_util.h"

namespace cinn {
namespace runtime {
namespace cuda {

CUDAModule::CUDAModule(const std::string& data, Kind kind) : data_(data), kind_(kind) {
  CHECK(!data.empty());

  cudaGetDeviceCount(&num_devices_);
  CHECK_GT(num_devices_, 0) << "No available devices";

  // TODO(Superjomn) Determine whether to initialize all the devices.
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaSetDevice(current_device_id);
  cuDeviceGet(&device_, current_device_id);
  cuCtxGetCurrent(&context_);
  cuDevicePrimaryCtxRetain(&context_, device_);
}

void CUDAModule::LaunchKernel(int device_id,
                              const std::string& func_name,
                              dim3 gridDim,
                              dim3 blockDim,
                              void** args,
                              size_t share_memory_size,
                              CUstream stream) {
  auto function = GetFunction(device_id, func_name);
  CHECK(function);
  CUDA_DRIVER_CALL(cuLaunchKernel(function,
                                  gridDim.x,
                                  gridDim.y,
                                  gridDim.z,
                                  blockDim.x,
                                  blockDim.y,
                                  blockDim.z,
                                  share_memory_size,
                                  stream,
                                  args,
                                  nullptr));
}

CUfunction CUDAModule::GetFunction(int device_id, const std::string& func_name) {
  VLOG(5) << "GetFuncion : " << func_name << " with device_id : " << device_id;
  if (!module_per_card_[device_id]) {
    std::lock_guard<std::mutex> lock(mutex_);
    CUDA_DRIVER_CALL(cuModuleLoadData(&module_per_card_[device_id], data_.c_str()));
  }

  CUfunction func;
  CUDA_DRIVER_CALL(cuModuleGetFunction(&func, module_per_card_[device_id], func_name.c_str()));
  return func;
}

CUdeviceptr CUDAModule::GetGlobal(int device_id, const std::string& name, size_t nbytes) {
  if (!module_per_card_[device_id]) {
    std::lock_guard<std::mutex> lock(mutex_);
    CUDA_DRIVER_CALL(cuModuleLoadData(&module_per_card_[device_id], data_.c_str()));
  }

  CUdeviceptr global;
  size_t _nbytes;
  CUDA_DRIVER_CALL(cuModuleGetGlobal(&global, &_nbytes, module_per_card_[device_id], name.c_str()));
  return global;
}

CUDAModule::~CUDAModule() {
  for (int i = 0; i < module_per_card_.size(); i++) {
    auto* module = module_per_card_[i];
    if (module) {
      CUDA_CALL(cudaSetDevice(i));
      CUDA_DRIVER_CALL(cuModuleUnload(module));
    }
  }
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn

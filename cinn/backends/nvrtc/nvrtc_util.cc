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

#include "cinn/backends/nvrtc/nvrtc_util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "cinn/backends/cuda_util.h"
#include "cinn/common/common.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace backends {
namespace nvrtc {
namespace {
#ifdef NVRTC_STL_PATH
static constexpr char* nvrtc_stl_path = NVRTC_STL_PATH;
#else
static constexpr char* nvrtc_stl_path = nullptr;
#endif
}  // namespace

std::string Compiler::operator()(const std::string& code, bool include_headers) {
  return CompilePTX(code, include_headers);
}

std::vector<std::string> Compiler::FindCUDAIncludePaths() {
  const std::string delimiter = "/";
  std::string cuda_include_path;
  const char* cuda_path_env = std::getenv("CUDA_PATH");
  if (cuda_path_env != nullptr) {
    cuda_include_path += cuda_path_env;
    cuda_include_path += delimiter + "include";
    return {cuda_include_path};
  }

#if defined(__linux__)
  struct stat st;
  cuda_include_path = "/usr/local/cuda/include";
  if (stat(cuda_include_path.c_str(), &st) == 0) {
    return {cuda_include_path};
  }
#endif
  LOG(FATAL) << "Cannot find cuda include path."
             << "CUDA_PATH is not set or CUDA is not installed in the default installation path."
             << "In other than linux, it is necessary to set CUDA_PATH.";
  return {cuda_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() { return {Context::Global().runtime_include_dir()}; }

std::string Compiler::CompilePTX(const std::string& code, bool include_headers) {
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;
  std::string cc = "30";
  int major, minor;
  cudaError_t e1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }

  compile_options.push_back("-arch=compute_" + cc);
  compile_options.push_back("-std=c++14");
  compile_options.push_back("-default-device");

  if (include_headers) {  // prepare include headers
    auto cuda_headers = FindCUDAIncludePaths();
    auto cinn_headers = FindCINNRuntimeIncludePaths();
    std::vector<std::string> include_paths;
    for (auto& header : cuda_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    for (auto& header : cinn_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    if (nvrtc_stl_path) {
      include_paths.push_back("--include-path=" + std::string{nvrtc_stl_path});
    }

    compile_options.insert(std::end(compile_options), include_paths.begin(), include_paths.end());
  }

  for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
  }
  VLOG(3) << "compile options: " << utils::Join(compile_options, " ");
  NVRTC_CALL(nvrtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  nvrtcResult compile_res = nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  {  // get log
    size_t log_size;
    NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
    CHECK_EQ(compile_res, NVRTC_SUCCESS) << log;
  }

  size_t ptx_size;
  NVRTC_CALL(nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CALL(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CALL(nvrtcDestroyProgram(&prog));

  return ptx;
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
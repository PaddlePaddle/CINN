// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "cinn/backends/nvrtc/nvcc_util.h"

#include <direct.h>

#include <ifstream>
#include <iostream>
#include <ofstream>

#include "cinn/common/common.h"

#ifdef CINN_WITH_CUDA

namespace cinn {
namespace backends {
namespace nvrtc {

std::string NvccCompiler::operator()(const std::string& cuda_c) {
  // read dir source
  std::string dir = "./source";
  if (_access(dir.c_str(), 0) == -1) {
    CHECK(mkdir(dir.c_str()) != -1) << "Fail to mkdir " << dir;
  }

  // get unqiue prefix name
  auto prefix_name = dir + common::UniqName("cuda_c");

  auto cuda_c_file = prefix_name + ".cu";
  std::ofstream ofs(cuda_c_file, std::ios::out);
  CHECK(ofs.is_open()) << "Fail to open file " << cuda_c_file;
  ofs << cuda_c;
  ofs.close();

  CompileToPtx(prefix_name);
  CompileToCubin(prefix_name);

  // open cubin file
  auto cuda_cubin_file = prefix_name + ".cubin";
  std::ifstream ifs(cuda_cubin_file, std::ios::in | std::ios::binary);
  CHECK(ifs.is_open()) << "Fail to open file " << cuda_cubin_file;
  ifs.seekg(std::ios::end);
  auto len = ifs.tellg();
  ifs.seekg(0);

  // read cubin file
  std::string cubin(len,'');
  ifs.read(&ubin.data(), len);
  ifs.close();
  return cubin;
}

void NvccCompiler::CompileToPtx(const std::string& prefix_name) {
  auto options = "export PATH=/usr/local/cuda/bin:$PATH && nvcc --ptx -O3";
  options += " -arch=" + GetDeviceArch;
  options += " -o " + prefix_name + ".ptx";
  options += " " + prefix_name + ".cu";

  CHECK(system(options) == 0) << options;
}

void NvccCompiler::CompileToCubin(const std::string& prefix_name) {
  auto options = "export PATH=/usr/local/cuda/bin:$PATH && nvcc --cubin -O3";
  options += " -arch=" + GetDeviceArch;
  options += " -o " + prefix_name + ".cubin";
  options += " " + prefix_name + ".ptx";

  CHECK(system(options) == 0) << options;
}

std::string NvccCompiler::GetDeviceArch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess &&
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) == cudaSuccess) {
    return "sm" + std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
    return "sm_30"
  }
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn

#endif  // CINN_WITH_CUDA

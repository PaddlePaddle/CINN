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
#pragma once

#ifdef CINN_WITH_CUDA

namespace cinn {
namespace backends {
namespace nvrtc {

class NvccCompiler {
 public:
  NvccCompiler(){};
  ~NvccCompiler(){};
  std::string operator()(const std::string&);
  std::string GetPtx();

 private:
  void CompileToPtx(const std::string&);
  void CompileToCubin(const std::string&);
  std::string GetDeviceArch();

  std::string ReadFile(const std::string&, std::ios_base::openmode);
};

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn

#endif  // CINN_WITH_CUDA

// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

class AccuracyChecker {
 public:
  AccuracyChecker(const Target& target,
                  Scope* scope,
                  const std::vector<std::string>& in_args,
                  const std::vector<std::string>& out_args)
      : target_(target), scope_(scope), in_args_({in_args}), out_args_({out_args}) {}

  bool operator()(std::map<std::string, bool>* out_args_check_result);
  bool operator()(const std::map<std::string, cinn_pod_value_t>& name2podargs,
                  std::map<std::string, bool>* out_args_check_result);

 private:
  template <typename T>
  Tensor CopyTensorToCpu(const Tensor& tensor);

  template <typename T>
  Tensor CopyBufferToCpu(const cinn_buffer_t* buffer);

  template <typename T>
  void MemcpyDeviceToHost(const T* src, size_t numel, T* dst);

  template <typename T>
  bool CheckNanOrInf(const Tensor& cpu_tensor);

  Target target_;
  Scope* scope_;  // Not owned
  std::vector<std::string> in_args_;
  std::vector<std::string> out_args_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

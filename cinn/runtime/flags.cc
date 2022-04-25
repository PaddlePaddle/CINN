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

#include "cinn/runtime/flags.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstdlib>
#include <string>

#ifdef CINN_WITH_CUDNN
DEFINE_bool(cinn_cudnn_deterministic,
            false,
            "Whether allow using an autotuning algorithm for convolution "
            "operator. The autotuning algorithm may be non-deterministic. If "
            "true, the algorithm is deterministic.");
#endif

DEFINE_bool(cinn_use_new_fusion_pass, false, "Whether use to new op_fusion and fusion_merge pass.");

namespace cinn {
namespace runtime {

void SetCinnCudnnDeterministic(bool state) {
#ifdef CINN_WITH_CUDNN
  FLAGS_cinn_cudnn_deterministic = state;
#else
  LOG(WARNING) << "CINN is compiled without cuDNN, this api is invalid!";
#endif
}

bool GetCinnCudnnDeterministic() {
#ifdef CINN_WITH_CUDNN
  return FLAGS_cinn_cudnn_deterministic;
#else
  LOG(FATAL) << "CINN is compiled without cuDNN, this api is invalid!";
  return false;
#endif
}

bool GetCinnUseNewFusionPassFromEnv() {
  const char *env_str = std::getenv("FLAGS_cinn_use_new_fusion_pass");
  if (env_str) {
    FLAGS_cinn_use_new_fusion_pass = static_cast<bool>(std::stoi(std::string(env_str)));
  }
  return FLAGS_cinn_use_new_fusion_pass;
}

}  // namespace runtime
}  // namespace cinn

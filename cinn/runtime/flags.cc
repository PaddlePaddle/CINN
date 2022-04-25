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

DEFINE_bool(cinn_use_new_fusion_pass, false, "Whether use the new op_fusion and fusion_merge pass.");
DEFINE_bool(cinn_sync_run, false, "Whether sync all devices after each instruction run, which is used for debug.");

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

bool ParseFromEnv(std::string name, bool default_value) {
  const char *env_str = std::getenv(name.c_str());
  if (env_str) {
    VLOG(4) << "Parse " << name << ", get " << env_str;
    return static_cast<bool>(std::stoi(std::string(env_str)));
  } else {
    return default_value;
  }
}

bool GetCinnUseNewFusionPassFromEnv() {
  static bool parsed = false;
  if (!parsed) {
    FLAGS_cinn_use_new_fusion_pass = ParseFromEnv("FLAGS_cinn_use_new_fusion_pass", FLAGS_cinn_use_new_fusion_pass);
    parsed                         = true;
  }
  return FLAGS_cinn_use_new_fusion_pass;
}

bool GetCinnSyncRunFromEnv() {
  static bool parsed = false;
  if (!parsed) {
    FLAGS_cinn_sync_run = ParseFromEnv("FLAGS_cinn_sync_run", FLAGS_cinn_sync_run);
    parsed              = true;
  }
  return FLAGS_cinn_sync_run;
}

}  // namespace runtime
}  // namespace cinn

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

#ifdef CINN_WITH_CUDNN
DEFINE_bool(cinn_cudnn_deterministic,
            false,
            "Whether allow using an autotuning algorithm for convolution "
            "operator. The autotuning algorithm may be non-deterministic. If "
            "true, the algorithm is deterministic.");
#endif

using ::GFLAGS_NAMESPACE::BoolFromEnv;
using ::GFLAGS_NAMESPACE::StringFromEnv;

// FLAGS to switch optimization status
DEFINE_bool(cinn_use_new_fusion_pass,
            BoolFromEnv("FLAGS_cinn_use_new_fusion_pass", false),
            "Whether use the new op_fusion and fusion_merge pass.");

DEFINE_bool(cinn_use_fill_constant_folding,
            BoolFromEnv("FLAGS_cinn_use_fill_constant_folding", false),
            "Whether use the FillConstantFolding pass.");

DEFINE_bool(cinn_use_cuda_vectorize,
            BoolFromEnv("FLAGS_cinn_use_cuda_vectorize", false),
            "Whether use cuda vectroize on schedule config");

// FLAGS for performance analysis and accuracy debug
DEFINE_bool(cinn_sync_run,
            BoolFromEnv("FLAGS_cinn_sync_run", false),
            "Whether sync all devices after each instruction run, which is used for debug.");

DEFINE_bool(cinn_self_check_accuracy,
            BoolFromEnv("FLAGS_cinn_self_check_accuracy", false),
            "Whether self-check accuracy after each instruction run, which is used for debug.");

DEFINE_string(cinn_fusion_groups_graphviz_dir,
              StringFromEnv("FLAGS_cinn_fusion_groups_graphviz_dir", ""),
              "Specify the directory path of dot file of graph, which is used for debug.");

DEFINE_string(cinn_source_code_save_path,
              StringFromEnv("FLAGS_cinn_source_code_save_path", ""),
              "Specify the directory path of generated source code, which is used for debug.");

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

}  // namespace runtime
}  // namespace cinn

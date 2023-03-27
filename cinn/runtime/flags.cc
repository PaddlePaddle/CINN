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
using ::GFLAGS_NAMESPACE::Int32FromEnv;
using ::GFLAGS_NAMESPACE::Int64FromEnv;
using ::GFLAGS_NAMESPACE::StringFromEnv;

DEFINE_string(cinn_x86_builtin_code_root, StringFromEnv("FLAGS_cinn_x86_builtin_code_root", ""), "");

DEFINE_int32(cinn_parallel_compile_size,
             // Revert changes in PR #990 to pass the model unittests
             Int32FromEnv("FLAGS_cinn_parallel_compile_size", 8),
             "When use parallel compile, set the number of group compiled by each thread.");

DEFINE_bool(cinn_use_op_fusion, BoolFromEnv("FLAGS_cinn_use_op_fusion", true), "Whether to use op fusion pass.");

DEFINE_bool(cinn_use_cublas_gemm, BoolFromEnv("FLAGS_cinn_use_cublas_gemm", true), "Whether to use cublas gemm.");

DEFINE_bool(cinn_use_common_subexpression_elimination,
            BoolFromEnv("FLAGS_cinn_use_common_subexpression_elimination", false),
            "Whether to use common subexpression elimination pass.");

DEFINE_string(cinn_custom_call_deny_ops,
              StringFromEnv("FLAGS_cinn_custom_call_deny_ops", ""),
              "a blacklist of op are denied by MarkCustomCallOps pass, separated by ;");

DEFINE_bool(cinn_use_custom_call,
            BoolFromEnv("FLAGS_cinn_use_custom_call", true),
            "Whether to use custom_call for ops with external_api registered");

DEFINE_bool(cinn_use_fill_constant_folding,
            BoolFromEnv("FLAGS_cinn_use_fill_constant_folding", false),
            "Whether use the FillConstantFolding pass.");

DEFINE_bool(cinn_check_fusion_accuracy_pass,
            BoolFromEnv("FLAGS_cinn_check_fusion_accuracy_pass", false),
            "Check the correct of fusion kernels, if the results not satisfied 'allclose(rtol=1e-05f, atol=1e-08f)', "
            "report error and exited.");

DEFINE_bool(cinn_use_cuda_vectorize,
            BoolFromEnv("FLAGS_cinn_use_cuda_vectorize", false),
            "Whether use cuda vectroize on schedule config");

DEFINE_bool(cinn_ir_schedule,
            BoolFromEnv("FLAGS_cinn_ir_schedule", true),
            "Whether use reconstructed schedule primitives.");

DEFINE_bool(use_reduce_split_pass, BoolFromEnv("FLAGS_use_reduce_split_pass", false), "Whether use reduce split pass.");

// FLAGS for performance analysis and accuracy debug
DEFINE_bool(cinn_sync_run,
            BoolFromEnv("FLAGS_cinn_sync_run", false),
            "Whether sync all devices after each instruction run, which is used for debug.");

DEFINE_string(cinn_self_check_accuracy,
              StringFromEnv("FLAGS_cinn_self_check_accuracy", ""),
              "Whether self-check accuracy after each instruction run, which is used for debug.");

DEFINE_int64(cinn_self_check_accuracy_num,
             Int64FromEnv("FLAGS_cinn_self_check_accuracy_num", 0L),
             "Set self-check accuracy print numel, which is used for debug.");

DEFINE_string(cinn_fusion_groups_graphviz_dir,
              StringFromEnv("FLAGS_cinn_fusion_groups_graphviz_dir", ""),
              "Specify the directory path of dot file of graph, which is used for debug.");

DEFINE_string(cinn_source_code_save_path,
              StringFromEnv("FLAGS_cinn_source_code_save_path", ""),
              "Specify the directory path of generated source code, which is used for debug.");

DEFINE_bool(enable_auto_tuner, BoolFromEnv("FLAGS_enable_auto_tuner", false), "Whether enable auto tuner.");

DEFINE_bool(auto_schedule_use_cost_model,
            BoolFromEnv("FLAGS_auto_schedule_use_cost_model", true),
            "Whether to use cost model in auto schedule, this is an on-developing flag and it will be removed when "
            "cost model is stable.");

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

unsigned long long RandomSeed::seed_ = 0ULL;

unsigned long long RandomSeed::GetOrSet(unsigned long long seed) {
  if (seed != 0ULL) {
    seed_ = seed;
  }
  return seed_;
}

unsigned long long RandomSeed::Clear() {
  auto old_seed = seed_;
  seed_         = 0ULL;
  return old_seed;
}

}  // namespace runtime
}  // namespace cinn

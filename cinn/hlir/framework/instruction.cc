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

#include "cinn/hlir/framework/instruction.h"

#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/accuracy_checker.h"
#include "cinn/utils/profiler.h"

DECLARE_bool(cinn_sync_run);
DECLARE_bool(cinn_self_check_accuracy);

namespace cinn {
namespace hlir {
namespace framework {

void Instruction::UpdateArgsCache(const std::map<std::string, cinn_pod_value_t>* name2podargs) {
  int cache_size = size();
  args_cached_.resize(cache_size);

  for (int i = 0; i < cache_size; ++i) {
    common::ArgsBuilder builder;
    // Remove duplicate input arguments
    std::unordered_set<std::string> in_args_set;
    std::vector<std::string> all_args;
    for (const auto& arg : in_args_[i]) {
      if (in_args_set.count(arg) != 0) continue;
      all_args.push_back(arg);
      in_args_set.insert(arg);
    }

    all_args.insert(std::end(all_args), out_args_[i].begin(), out_args_[i].end());

    if (name2podargs != nullptr) {
      for (const auto& arg : all_args) {
        CHECK_NE(name2podargs->count(arg), 0) << "Argument [" << arg << "] not found in the name2podargs";
        builder.Add(name2podargs->at(arg));
      }
    } else {
      for (const auto& arg : all_args) {
        auto* var = scope_->FindVar(arg);
        CHECK(var) << "Argument [" << arg << "] not found in the scope";

        // TODO(Superjomn) Support other types.
        auto& tensor = absl::get<Tensor>(*var);
        builder.Add(tensor->buffer());
      }
    }

    args_cached_[i] = builder.Build();
  }
}

void Instruction::Finalize() {
  if (fn_ptrs_.size() > 1 && fn_ptrs_.size() != in_args_.size()) {
    out_args_.back()[0] = out_args_.front()[0];
    out_args_.erase(out_args_.begin());
    in_args_.erase(in_args_.begin());
  }

  finalized_flag_ = true;
}

void Instruction::Run(const std::map<std::string, cinn_pod_value_t>* name2podargs,
                      bool dryrun,
                      void* stream,
                      bool use_cache) {
  utils::RecordEvent record_run(function_name_);
  CHECK(finalized_flag_) << "Instruction must be finalized before run";
  if (function_name_ == "no_run") {
    VLOG(2) << "skip instruction";
    return;
  }

  VLOG(2) << "Run function " << function_name_;

  {
    utils::RecordEvent record_args("PrepareArgs");
    if (!use_cache || args_cached_.size() != size()) {
      UpdateArgsCache(name2podargs);
    }
  }

  utils::ProfilerRangePush("Compute");
  for (int idx = 0; idx < fn_ptrs_.size(); ++idx) {
    VLOG(6) << "Runing func name: " << fn_names_[idx];
    auto& pod_args = args_cached_[idx];
    CHECK(fn_ptrs_[idx]) << "The LoweredFunc address should be set first by calling SetLoweredFunc method";
    if (!dryrun) {
      if (target_ == common::DefaultNVGPUTarget()) {
        ((lower_func_ptr_g)fn_ptrs_[idx])(static_cast<void*>(pod_args.data()), pod_args.size(), stream);
      } else {
        ((lower_func_ptr_t)fn_ptrs_[idx])(static_cast<void*>(pod_args.data()), pod_args.size());
      }
    }
  }
  utils::ProfilerRangePop();

  if (FLAGS_cinn_self_check_accuracy) {
    CheckResults(name2podargs, stream);
#ifdef CINN_WITH_CUDA
  } else if (FLAGS_cinn_sync_run) {
    utils::RecordEvent record_sync("Synchronize");
    auto st = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    if (st) {
      LOG(FATAL) << "cuda error -> " << cudaGetErrorString(st);
    }
#endif
  }
}

void Instruction::CheckResults(const std::map<std::string, cinn_pod_value_t>* name2podargs, void* stream) {
#ifdef CINN_WITH_CUDA
  cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
#endif

  if (fn_names_.size() == 1) {
    std::unordered_set<std::string> skipped_instr_set = {"malloc_buffer_instruction", "free_buffer_instruction"};
    for (auto& name : skipped_instr_set) {
      if (fn_names_[0].find(name) != std::string::npos) {
        // Skip the malloc & free buffer instructions.
        return;
      }
    }
  }

  AccuracyChecker checker(target_, scope_);

  LOG(WARNING) << "Instruction {";
  for (size_t i = 0; i < fn_names_.size(); ++i) {
    LOG(WARNING) << "  Function " << fn_names_[i] << ":";
    for (auto& in_name : in_args_[i]) {
      std::string result_str;
      if (name2podargs) {
        result_str = checker(name2podargs, in_name);
      } else {
        result_str = checker(in_name);
      }
      LOG(WARNING) << "    input: " << result_str;
    }
    for (auto& out_name : out_args_[i]) {
      std::string result_str;
      if (name2podargs) {
        result_str = checker(name2podargs, out_name);
      } else {
        result_str = checker(out_name);
      }
      LOG(WARNING) << "    output: " << result_str;
    }
  }
  LOG(WARNING) << "}";
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

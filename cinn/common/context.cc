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

#include "cinn/common/context.h"

#include <glog/logging.h>
#include <isl/cpp.h>

#include <mutex>

#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

thread_local isl::ctx Context::ctx_ = isl_ctx_alloc();
thread_local InfoRegistry Context::info_rgt_;
thread_local DebugManager Context::debug_mgr_;

Context& Context::Global() {
  static Context x;
  isl_options_set_on_error(ctx_.get(), ISL_ON_ERROR_ABORT);
  return x;
}

const std::string& Context::runtime_include_dir() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (runtime_include_dir_.empty()) {
    char* env            = std::getenv(kRuntimeIncludeDirEnvironKey);
    runtime_include_dir_ = env ? env : "";  // Leave empty if no env found.
  }
  VLOG(4) << "-- runtime_include_dir: " << runtime_include_dir_;
  return runtime_include_dir_;
}

const char* kRuntimeIncludeDirEnvironKey = "runtime_include_dir";

std::string NameGenerator::New(const std::string& name_hint) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = name_hint_idx_.find(name_hint);
  if (it == name_hint_idx_.end()) {
    name_hint_idx_.emplace(name_hint, -1);
    return name_hint;
  }
  return name_hint + "_" + std::to_string(++it->second);
}

}  // namespace common

DEFINE_bool(cinn_runtime_display_debug_info, false, "Whether to display debug information in runtime");
}  // namespace cinn

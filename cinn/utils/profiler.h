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

#include <functional>
#include <string>

#ifdef CINN_WITH_NVTX
#include <nvToolsExt.h>
#endif

#include "cinn/utils/event.h"

namespace cinn {
namespace utils {

enum class ProfilerState {
  kDisabled,  // disabled state
  kCPU,       // CPU profiling state
  kCUDA,      // GPU profiling state
  kAll
};

class ProfilerHelper {
 public:
  static ProfilerState g_state;

  static void Enable() { g_state = ProfilerState::kAll; }
  static void Enable_CPU() { g_state = ProfilerState::kCPU; }
  static void Enable_CUDA() { g_state = ProfilerState::kCUDA; }
};

class RecordEvent {
  using CallBack = std::function<void()>;

 public:
  RecordEvent(const std::string& name, EventType type = EventType::kOrdinary);

  void End();

  ~RecordEvent() { End(); }

 private:
  CallBack call_back_;
};

void SynchronizeAllDevice();

void ProfilerStart();

void ProfilerStop();

void ProfilerRangePush(const std::string& name);

void ProfilerRangePop();

inline bool is_profiler_enable() { return ProfilerHelper::g_state != ProfilerState::kDisabled; }

inline bool is_cpu_enable() {
  return ProfilerHelper::g_state == ProfilerState::kAll || ProfilerHelper::g_state == ProfilerState::kCPU;
}

inline bool is_cuda_enable() {
  return ProfilerHelper::g_state == ProfilerState::kAll || ProfilerHelper::g_state == ProfilerState::kCUDA;
}

}  // namespace utils
}  // namespace cinn

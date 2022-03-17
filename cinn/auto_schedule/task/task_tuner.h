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

#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/auto_schedule/task/tuning_options.h"

namespace cinn {
namespace auto_schedule {

// This class is responsible for tuning a specific task,
// it will integrate necessary components to search the
// optimal schedule for the task.
class TaskTuner {
 public:
  TaskTuner(const TuneTask& task) : task_(&task) {}

  void Tune(const TuningOptions& options);

 private:
  const TuneTask* task_;
};

}  // namespace auto_schedule
}  // namespace cinn

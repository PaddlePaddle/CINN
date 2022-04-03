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

#include <memory>

#include "cinn/auto_schedule/measure/schedule_measurer.h"
#include "cinn/auto_schedule/search_strategy/evolutionary_search.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/auto_schedule/tuning.h"

namespace cinn {
namespace auto_schedule {

// This class is responsible for tuning a specific task,
// it will integrate necessary components to search the
// optimal schedule for the task.
class TaskOptimizer {
 public:
  TaskOptimizer(const TuneTask& task, ScheduleMeasurer* schedule_measure)
      : task_(&task), schedule_measure_(schedule_measure) {}

  TuningResult::OptimizedLoweredFuncs Optimize(const TuningOptions& options);

 private:
  TuningResult::OptimizedLoweredFuncs OptimizeByEvolution(const TuningOptions& options);

  const TuneTask* task_;

  ScheduleMeasurer* schedule_measure_;

  std::unique_ptr<EvolutionarySearch> = nullptr;
};

}  // namespace auto_schedule
}  // namespace cinn

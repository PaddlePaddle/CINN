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

#include "cinn/auto_schedule/task_scheduler/task_scheduler.h"

#include <algorithm>

#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/auto_schedule/task_scheduler/efficiency_priority.h"
#include "cinn/auto_schedule/task_scheduler/round_robin.h"

namespace cinn {
namespace auto_schedule {

std::shared_ptr<TaskScheduler> TaskScheduler::Make(const std::vector<TuneTask>& tasks,
                                                   const Config& config,
                                                   const std::string& strategy) {
  if (strategy == "round_robin") {
    return std::make_shared<RoundRobin>(tasks, config);
  } else if (strategy == "efficiency_priority") {
    return std::make_shared<EfficiencyPriority>(tasks, config);
  }

  LOG(FATAL) << "Unimplementd strategy:" << strategy;
  return nullptr;
}

TaskScheduler::TaskScheduler(const std::vector<TuneTask>& tasks, const Config& config)
    : tasks_(&tasks), config_(config), cur_task_id_(0) {
  CHECK_GT(tasks.size(), 0) << "Empty task";
  CHECK_GT(config.num_tuning_rounds, 0) << "Invalid config";

  task_tuners_.resize(tasks.size());
  std::transform(tasks.begin(), tasks.end(), task_tuners_.begin(), [](const TuneTask& task) {
    return std::make_unique<TaskTuner>(task);
  });
}

void TaskScheduler::Run(const TuningOptions& tune_options) {
  // TODO(CtfGo): support tuners execute in synchronous
  // or asynchronous parallel
  for (int r = 0; r < config_.num_tuning_rounds; ++r) {
    this->cur_task_id_ = 0;
    int run_id         = -1;
    while ((run_id = NextTaskId()) != -1) {
      auto* tuner = task_tuners_.at(run_id).get();
      tuner->Tune(tune_options);
    }
  }
}

}  // namespace auto_schedule
}  // namespace cinn

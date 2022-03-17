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
#include <memory>
#include <string>
#include <vector>

#include "cinn/auto_schedule/task/task_tuner.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/auto_schedule/task/tuning_options.h"

namespace cinn {
namespace auto_schedule {

// Class for scheduling tasks to perform auto-tune
class TaskScheduler {
 public:
  // All configs of different schedule strategies
  // needed will be defined here together.
  struct Config {
    // The number of tuning rounds, each round will
    // involve TuningOptions.num_measure_trials measurements.
    int num_tuning_rounds = 1;
    // The minimum threshold of earnings ratio, used by EfficiencyPriority
    float minimum_gain_threshold = 0.0;
  };

  // Create a TaskScheduler with the specific strategy name
  // and necessary construct parameters.
  static std::shared_ptr<TaskScheduler> Make(const std::vector<TuneTask>& tasks,
                                             const Config& config,
                                             const std::string& strategy = "round_robin");

  // Return the name of schedule strategy
  virtual const char* Name() const = 0;

  // Schedule all tunners to tune tasks
  void Run(const TuningOptions& tune_options);

 protected:
  // A taskScheduler object should be created with the static function Make
  TaskScheduler(const std::vector<TuneTask>& tasks, const Config& config);

  // Select a task to tune
  virtual int NextTaskId() = 0;

  // The pointer refers to all tasks
  const std::vector<TuneTask>* tasks_;
  // The config for scheduling strategy
  Config config_;
  // The current task id to be estimated
  int cur_task_id_;
  // The tuners to perform auto-tune, each tuner assigns a task.
  std::vector<std::unique_ptr<TaskTuner>> task_tuners_;
};

}  // namespace auto_schedule
}  // namespace cinn

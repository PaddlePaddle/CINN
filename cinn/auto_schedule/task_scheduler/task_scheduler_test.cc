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

#include <gtest/gtest.h>

#include <type_traits>

#include "cinn/auto_schedule/task_scheduler/efficiency_priority.h"
#include "cinn/auto_schedule/task_scheduler/round_robin.h"

namespace cinn {
namespace auto_schedule {

TEST(TaskScheduler, Make) {
  std::vector<TuneTask> tasks(3);
  TaskScheduler::Config config;

  auto round_robin = TaskScheduler::Make(tasks, config);
  ASSERT_STREQ(round_robin->Name(), "round_robin");
  auto efficiency_priority = TaskScheduler::Make(tasks, config, "efficiency_priority");
  ASSERT_STREQ(efficiency_priority->Name(), "efficiency_priority");
}

// TODO(CtfGo): Add tests for member functions (Run, NextTaskId) of
// a detail TaskScheduler object after we define
// the callback for returning tuning results in TuningOptions
TEST(TaskScheduler, Run) {}

TEST(RoundRobinScheduler, NextTaskId) {}

TEST(EfficiencyPriorityScheduler, NextTaskId) {}

}  // namespace auto_schedule
}  // namespace cinn

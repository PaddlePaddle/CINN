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

#include "cinn/auto_schedule/auto_tuner.h"

#include <algorithm>

#include "cinn/auto_schedule/task/task_creator.h"
#include "cinn/auto_schedule/task_scheduler/task_scheduler.h"

namespace cinn {
namespace auto_schedule {

AutoTuner::AutoTuner(const common::Target& target, hlir::framework::Graph* graph) : target_(target), graph_(graph) {}

void AutoTuner::Initialize(const Config& config, hlir::framework::GraphCompiler* graph_compiler) {
  // create tasks
  TaskCreator task_creator;
  tasks_ = task_creator.CreateTuneTaskOpLevel(graph_);
  CHECK_GT(tasks_.size(), 0) << "Create tasks failed";

  // create task optimizers
  task_optimizers_.resize(tasks_.size());
  std::transform(tasks_.begin(), tasks_.end(), task_optimizers_.begin(), [](const TuneTask& task) {
    return std::make_unique<TaskOptimizer>(task);
  });

  // create task scheduler
  task_scheduler_ = TaskScheduler::Make(tasks_, config.task_schedule_config, config.task_schedule_strategy);
}

TuningResult AutoTuner::Tune(const TuningOptions& options) {
  CHECK_GT(options.num_tuning_rounds, 0) << "Invalid config";

  TuningResult result;
  result.tuned_graphs.resize(tasks_.size());
  result.optimized_schedules.resize(tasks_.size());
  // A task only tunes schedule now, so we populate its sub_graph
  // as default result of graph tuning, and that should be updated
  // once we support graph tuning.
  for (auto i = 0; i < tasks_.size(); ++i) {
    auto&& task                   = tasks_.at(i);
    result.tuned_graphs[i].groups = task.task_graph();
  }

  for (int r = 0; r < options.num_tuning_rounds; ++r) {
    int run_id = -1;
    task_scheduler_->Reset();
    while ((run_id = task_scheduler_->NextTaskId()) != -1) {
      auto* opt               = task_optimizers_.at(run_id).get();
      auto optimized_schedule = opt->optimize(options);
      // update the best schedules searched so far.
      result.optimized_schedules.at(run_id) = optimized_schedule;
    }
  }

  return result;
}

}  // namespace auto_schedule
}  // namespace cinn

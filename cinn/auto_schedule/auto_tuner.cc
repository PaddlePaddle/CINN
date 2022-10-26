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

#include <glog/logging.h>
#include <pybind11/embed.h>

#include <algorithm>
#include <memory>
#include <utility>

#include "cinn/auto_schedule/database/jsonfile_database.h"
#include "cinn/auto_schedule/measure/schedule_measurer.h"
#include "cinn/auto_schedule/measure/simple_builder.h"
#include "cinn/auto_schedule/measure/simple_runner.h"
#include "cinn/auto_schedule/task/task_creator.h"
#include "cinn/auto_schedule/task/task_registry.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/auto_schedule/task_scheduler/task_scheduler.h"
#include "cinn/common/context.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/visualize_helper.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

AutoTuner::AutoTuner(const common::Target& target, hlir::framework::Graph* graph) : target_(target), graph_(graph) {}

void AutoTuner::Initialize(const Config& config, hlir::framework::GraphCompiler* graph_compiler) {
  // create builder, runner, and schedule measurer
  builder_           = std::make_unique<SimpleBuilder>(graph_compiler);
  runner_            = std::make_unique<SimpleRunner>(config.runner_repeat_times);
  schedule_measurer_ = std::make_unique<ScheduleMeasurer>(builder_.get(), runner_.get());

  // initialize database
  database_ = std::move(Database::Make(config.database_config));

  // create tasks
  TaskCreator task_creator;
  tasks_ = task_creator.CreateTuneTaskOpLevel(graph_);

  const auto& dtype_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, common::Type>>("inferdtype");
  const auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");

  op_lowerer_                        = std::make_unique<hlir::framework::OpLowerer>(dtype_dict, shape_dict, target_);
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  for (auto i = 0; i < tasks_.size(); ++i) {
    auto&& task = tasks_[i];
    task.SetOpLowerer(op_lowerer_.get());
    task.TaskGraphToUnoptLoweredFunc();
    task.SerializeToString(shape_dict, dtype_dict);

    // Register the initial ModuleExpr corresponding to the task
    std::vector<ir::Expr> exprs(task.lowered_funcs.size());
    std::transform(
        task.lowered_funcs.begin(), task.lowered_funcs.end(), exprs.begin(), [](const ir::LoweredFunc& func) {
          return func->body;
        });
    task_registry->Regist(task.serialized_key, ir::ModuleExpr(exprs));

    VLOG(3) << "Add a task, id:" << i << ", serialized_key:\n" << task.serialized_key;
  }

  // create task optimizers
  task_optimizers_.resize(tasks_.size());
  std::transform(tasks_.begin(), tasks_.end(), task_optimizers_.begin(), [&](const TuneTask& task) {
    return std::make_unique<TaskOptimizer>(task, schedule_measurer_.get(), database_.get());
  });

  // create task scheduler
  task_scheduler_ = TaskScheduler::Make(tasks_, config.task_schedule_config, config.task_schedule_strategy);
}

void PrintResult(const TuningResult::TunedSubGraph& sub_graph) {
  if (!VLOG_IS_ON(3)) {
    return;
  }

  VLOG(3) << "Group size of sub graph:" << sub_graph.groups.size();
  for (auto i = 0; i < sub_graph.groups.size(); ++i) {
    const auto& group = sub_graph.groups.at(i)->CollectNodes();
    VLOG(3) << "Group-" << i << " node size:" << group.size();
    VLOG(3) << "Group " << i << " {";
    for (auto* node : group) {
      VLOG(3) << "  " << hlir::framework::DebugString(node);
    }
    VLOG(3) << "}";
  }
}

void PrintResult(const TuningResult::OptimizedComputeExpr& optimized_expr) {
  if (!VLOG_IS_ON(3)) {
    return;
  }

  VLOG(3) << "Group size of lowered function:" << optimized_expr.lowered_funcs.size();
  for (auto i = 0; i < optimized_expr.lowered_funcs.size(); ++i) {
    const auto& lowered_group = optimized_expr.lowered_funcs.at(i);
    VLOG(3) << "Lowered Group-" << i << " function size:" << lowered_group.size();
    for (auto j = 0; j < lowered_group.size(); ++j) {
      const ir::LoweredFunc& func = lowered_group.at(i);
      VLOG(3) << "LoweredFunc-" << j << " detail:\n" << func;
    }
  }
}

void PrintResult(const TuningResult& result) {
  if (!VLOG_IS_ON(3)) {
    return;
  }
  VLOG(3) << "###### Debug TuningResult ######\n";
  VLOG(3) << "Tuned SubGraph num:" << result.tuned_graph.size();
  for (auto i = 0; i < result.tuned_graph.size(); ++i) {
    VLOG(3) << "****** SubGraph-" << i << " Detail ******\n";
    PrintResult(result.tuned_graph.at(i));
    VLOG(3) << "****** SubGraph End ******";
  }

  VLOG(3) << "OptimizedComputeExpr num:" << result.optimized_exprs.size();
  for (auto i = 0; i < result.optimized_exprs.size(); ++i) {
    VLOG(3) << "****** OptimizedComputeExpr-" << i << " Detail ******\n";
    PrintResult(result.optimized_exprs.at(i));
    VLOG(3) << "****** OptimizedComputeExpr End ******";
  }
  VLOG(3) << "###### TuningResult End ######";
}

TuningResult AutoTuner::Tune(const TuningOptions& options) {
  CHECK_GT(options.num_tuning_rounds, 0) << "Invalid config";
  VLOG(3) << "Begin tuning with round num=" << options.num_tuning_rounds << ", tasks size=" << tasks_.size();

  TuningResult result;
  result.tuned_graph.resize(tasks_.size());
  result.optimized_exprs.resize(tasks_.size());
  // A task only tunes schedule now, so we populate its sub_graph
  // as default result of graph tuning, and that should be updated
  // once we support graph tuning.
  for (auto i = 0; i < tasks_.size(); ++i) {
    auto&& task                  = tasks_.at(i);
    result.tuned_graph[i].groups = task.task_graph;
  }

  for (int r = 0; r < options.num_tuning_rounds; ++r) {
    VLOG(3) << "<<<<<< Round " << r << " >>>>>>";
    int run_id = -1;
    task_scheduler_->Reset();
    while ((run_id = task_scheduler_->NextTaskId()) != -1) {
      VLOG(3) << "Start tuning task id:" << run_id;
      auto* opt           = task_optimizers_.at(run_id).get();
      auto optimized_expr = opt->Optimize(options);
      VLOG(3) << "Task finished, print optimized Expr:\n";
      PrintResult(optimized_expr);
      // update the best schedules searched so far.
      result.optimized_exprs.at(run_id) = std::move(optimized_expr);
    }
  }

  PrintResult(result);
  return result;
}

}  // namespace auto_schedule
}  // namespace cinn

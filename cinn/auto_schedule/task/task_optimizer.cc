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

#include "cinn/auto_schedule/task/task_optimizer.h"

#include <glog/logging.h>

#include <limits>

#include "cinn/auto_schedule/cost_model/expr_cost_model.h"
#include "cinn/auto_schedule/measure/measure.h"
#include "cinn/auto_schedule/search_strategy/evolutionary_search.h"
#include "cinn/common/target.h"
#include "cinn/hlir/framework/op_lowering.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/transform_gpu_forloop.h"
#include "cinn/runtime/flags.h"

DECLARE_bool(auto_schedule_use_cost_model);

namespace cinn {
namespace auto_schedule {

TaskOptimizer::TaskOptimizer(const TuneTask& task, ScheduleMeasurer* schedule_measurer)
    : task_(&task), schedule_measurer_(schedule_measurer), cost_model_() {}

TuningResult::OptimizedComputeExpr TaskOptimizer::Optimize(const TuningOptions& options) {
  // TODO(zhhsplendid): develop other optimize methods and configure the method by options.
  return OptimizeByEvolution(options);
}

TuningResult::OptimizedComputeExpr TaskOptimizer::OptimizeByEvolution(const TuningOptions& options) {
  CHECK_EQ(options.num_measure_trials % options.num_samples_per_iteration, 0)
      << "TuningOptions.num_measure_trials % TuningOptions.num_samples_per_iteration must be 0.";

  VLOG(4) << "TuneTask LoweredFunc before optimization is:";
  VLOG(4) << "task_.lowered_funcs().size() = " << task_->lowered_funcs.size();
  for (size_t i = 0; i < task_->lowered_funcs.size(); ++i) {
    VLOG(4) << "lowered_funcs[" << i << "] = ";
    VLOG(4) << task_->lowered_funcs[i];
  }

  if (evolutionary_search_ == nullptr) {
    // TODO(zhhsplendid): check whether the options is same as previous,
    // if not, we should create new EvolutionarySearch
    evolutionary_search_ = std::make_unique<EvolutionarySearch>(*task_, cost_model_);
  }

  if (options.num_measure_trials == 0) {
    std::vector<SearchState> states = evolutionary_search_->SearchModuleExprEpsGreedy(options);
    VLOG(4) << "TaskOptimizer run EvolutionarySearch with return size = " << states.size();
    TuningResult::OptimizedComputeExpr result;
    // TODO(zhhsplendid): current a task only contains one Op or one Fused Op,
    // so we can take only first std::vector<ir::LoweredFunc>. Support the
    // lowered_funcs to be std::vector<std::vector<ir::LoweredFunc>>
    // in the future.

    result.lowered_funcs.emplace_back(optim::IRCopy(task_->lowered_funcs));

    std::vector<ir::Expr> best_exprs = states[0].mod_expr.GetExprs();
    CHECK_EQ(best_exprs.size(), result.lowered_funcs[0].size())
        << "RuntimeError: Expr size is not equal to LoweredFunc size in TaskOptimizer";
    for (size_t i = 0; i < best_exprs.size(); ++i) {
      result.lowered_funcs[0][i] = FuncWithUpdatedBody(result.lowered_funcs[0][i], best_exprs[i]);
      if (task_->target == common::DefaultNVGPUTarget()) {
        result.lowered_funcs[0][i]->PrepareCudaAxisInfoFromBody();
      }
    }
    return result;
  }

  int measured_count   = 0;
  double min_exec_time = std::numeric_limits<double>().max();
  TuningResult::OptimizedComputeExpr result;
  result.lowered_funcs.push_back(optim::IRCopy(task_->lowered_funcs));

  while (measured_count < options.num_measure_trials) {
    std::vector<SearchState> states = evolutionary_search_->SearchModuleExprEpsGreedy(options);
    VLOG(4) << "TaskOptimizer run EvolutionarySearch with return size = " << states.size();
    std::vector<MeasureInput> measure_inputs(states.size());
    std::vector<const ir::ModuleExpr*> cost_model_samples(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
      cost_model_samples[i]            = &(states[i].mod_expr);
      measure_inputs[i].task           = task_;
      std::vector<ir::Expr> best_exprs = states[i].mod_expr.GetExprs();
      CHECK_EQ(best_exprs.size(), task_->lowered_funcs.size())
          << "RuntimeError: Expr size is not equal to LoweredFunc size in TaskOptimizer";

      measure_inputs[i].lowered_funcs.emplace_back(optim::IRCopy(task_->lowered_funcs));
      for (size_t j = 0; j < best_exprs.size(); ++j) {
        measure_inputs[i].lowered_funcs.front().at(j) =
            FuncWithUpdatedBody(measure_inputs[i].lowered_funcs.front().at(j), best_exprs[j]);
        if (task_->target == common::DefaultNVGPUTarget()) {
          measure_inputs[i].lowered_funcs.front().at(j)->PrepareCudaAxisInfoFromBody();
        }
      }
    }
    std::vector<MeasureResult> measure_outputs = schedule_measurer_->Measure(measure_inputs);
    CHECK_EQ(measure_outputs.size(), states.size())
        << "ScheduleMeasurer didn't output same number of MeasureOutput of states in TaskOptimizer";

    std::vector<float> cost_model_labels(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
      cost_model_labels[i] = measure_outputs[i].execution_cost;
    }

    if (FLAGS_auto_schedule_use_cost_model) {
      VLOG(6) << "cost_model_samples.size() = " << cost_model_samples.size();
      VLOG(6) << "cost_model_labels.size() = " << cost_model_labels.size();
      cost_model_.Update(cost_model_samples, cost_model_labels, task_->target);
    }
    // TODO(zhhsplendid): write measure record into cache.

    for (size_t i = 0; i < measure_outputs.size(); ++i) {
      if (measure_outputs[i].execution_cost < min_exec_time) {
        min_exec_time        = measure_outputs[i].execution_cost;
        result.lowered_funcs = measure_inputs[i].lowered_funcs;
      }
    }

    measured_count += states.size();
  }
  return result;
}

ir::LoweredFunc TaskOptimizer::FuncWithUpdatedBody(const ir::LoweredFunc& old_func, ir::Expr& body) {
  ir::ModuleExpr mod_expr(std::vector<ir::Expr>({body}));
  ir::IRSchedule ir_sch(mod_expr);

  // temp_bufs may be deleted during auto tuning (such as auto inline),
  // we have to check from old temp bufs and set them as local buffer.
  for (const ir::Buffer& buf : old_func->temp_bufs) {
    const std::string& buf_name              = buf->name;
    std::vector<ir::Expr> all_block_realizes = ir_sch.GetAllBlocks();
    for (ir::Expr& e : all_block_realizes) {
      const ir::ScheduleBlockRealize* sche_block_realize = e.As<ir::ScheduleBlockRealize>();
      const std::string& sche_name = sche_block_realize->schedule_block.As<ir::ScheduleBlock>()->name;
      if (buf_name == "_" + sche_name) {
        VLOG(6) << "Set local buffer for temp buffer " << buf_name;
        ir_sch.SetBuffer(e, "local", true);
        break;
      }
    }
  }

  ir::Expr updated_body = ir_sch.GetModule().GetExprs()[0];
#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&updated_body);
#endif
  // Get new temp bufs by analyzing.
  std::vector<ir::Buffer> new_temp_bufs = lang::GetTempBuffers(old_func->args, updated_body);
  ir::LoweredFunc new_func = ir::_LoweredFunc_::Make(old_func->name, old_func->args, updated_body, new_temp_bufs);
  new_func->PrepareBufferCastExprs();
  new_func = optim::Optimize(Expr(new_func), task_->target, false).as_lowered_func_ref();
  return new_func;
}

}  // namespace auto_schedule
}  // namespace cinn

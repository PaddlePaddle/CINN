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

#include <functional>
#include <limits>

#include "cinn/auto_schedule/analysis/analyze_ir.h"
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
#include "cinn/utils/string.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime_api.h>

#include "cinn/backends/cuda_util.h"
#endif

DECLARE_bool(auto_schedule_use_cost_model);

namespace cinn {
namespace auto_schedule {

TaskOptimizer::TaskOptimizer(const TuneTask& task, ScheduleMeasurer* schedule_measurer, Database* database)
    : task_(&task), schedule_measurer_(schedule_measurer), database_(database), cost_model_() {}

FunctionGroup TaskOptimizer::Optimize(const TuningOptions& options) {
  // TODO(zhhsplendid): develop other optimize methods and configure the method by options.
  return OptimizeByEvolution(options);
}

FunctionGroup TaskOptimizer::OptimizeByEvolution(const TuningOptions& options) {
  CHECK_EQ(options.num_measure_trials % options.num_samples_per_iteration, 0)
      << "TuningOptions.num_measure_trials % TuningOptions.num_samples_per_iteration must be 0.";

  VLOG(4) << "Optimizing TuneTask with num_measure_trials:" << options.num_measure_trials
          << ", LoweredFunc before optimization is:";
  VLOG(4) << "lowered function size = " << task_->lowered_funcs.size();
  for (size_t i = 0; i < task_->lowered_funcs.size(); ++i) {
    VLOG(4) << "lowered_funcs[" << i << "] detail:\n" << task_->lowered_funcs[i];
  }

  if (evolutionary_search_ == nullptr) {
    // TODO(zhhsplendid): check whether the options is same as previous,
    // if not, we should create new EvolutionarySearch
    evolutionary_search_ = std::make_unique<EvolutionarySearch>(*task_, cost_model_, database_);
  }

  // use initial lowered function as default result
  FunctionGroup result = optim::IRCopy(task_->lowered_funcs);
  if (options.num_measure_trials == 0) {  // no need to measure and simply return the best searched
    std::vector<MeasureInput> measure_candidates;
    std::vector<SearchState> states = SearchOneRound(options, &measure_candidates);
    if (!states.empty()) {
      result = measure_candidates[0].lowered_funcs;
    } else {
      LOG(WARNING) << "No valid candidate searched, will return initial state";
    }
    return result;
  }

  int measured_count            = 0;
  uint32_t continuous_empty_cnt = 0;
  double min_exec_time          = std::numeric_limits<double>().max();
  while (measured_count < options.num_measure_trials) {
    VLOG(4) << "Launch a new search, current measured_count:" << measured_count;
    std::vector<MeasureInput> measure_inputs;
    std::vector<SearchState> states = SearchOneRound(options, &measure_inputs);
    if (states.empty()) {  // no new valid candidate achieved
      ++continuous_empty_cnt;
      if (continuous_empty_cnt <= kMaxRetryContinuousEmpty_) {
        VLOG(4) << "No valid state searched, continuous_empty_cnt=" << continuous_empty_cnt;
        continue;
      } else {
        LOG(WARNING)
            << "OptimizeByEvolution will be exited in advance due to continuous invalid search, final measured_count="
            << measured_count;
        break;
      }
    }
    continuous_empty_cnt = 0;  // reset if get valid candidates

    VLOG(4) << "ScheduleMeasurer start with input size=" << measure_inputs.size();
    std::vector<MeasureResult> measure_outputs = schedule_measurer_->Measure(measure_inputs);
    CHECK_EQ(measure_outputs.size(), states.size())
        << "ScheduleMeasurer didn't output same number of MeasureOutput of states in TaskOptimizer";
    // record to database
    for (size_t i = 0; i < states.size(); ++i) {
      database_->AddRecord(
          TuningRecord(measure_inputs[i].task->serialized_key, states[i], measure_outputs[i].execution_cost));
    }

    // update cost model
    if (FLAGS_auto_schedule_use_cost_model) {
      std::vector<const ir::ModuleExpr*> cost_model_samples(states.size());
      std::vector<float> cost_model_labels(states.size());
      for (size_t i = 0; i < states.size(); ++i) {
        cost_model_samples[i] = &(states[i]->ir_schedule.GetModule());
        cost_model_labels[i]  = measure_outputs[i].execution_cost;
      }
      VLOG(4) << utils::StringFormat("Update CostModel with samples size=%lu,labels size=%lu",
                                     cost_model_samples.size(),
                                     cost_model_labels.size());
      cost_model_.Update(cost_model_samples, cost_model_labels, task_->target);
    }

    // update the best
    for (size_t i = 0; i < measure_outputs.size(); ++i) {
      if (measure_outputs[i].execution_cost < min_exec_time) {
        VLOG(4) << "Update best candidate with execution_cost:" << measure_outputs[i].execution_cost << "us";
        min_exec_time = measure_outputs[i].execution_cost;
        result        = measure_inputs[i].lowered_funcs;
      }
    }

    // count result size
    measured_count += states.size();
  }
  return result;
}

std::vector<SearchState> TaskOptimizer::SearchOneRound(const TuningOptions& options,
                                                       std::vector<MeasureInput>* measure_candidates) {
  std::vector<SearchState> states = evolutionary_search_->SearchModuleExprEpsGreedy(options);
  VLOG(4) << JoinStatesDebugString("TaskOptimizer::EvolutionarySearch-Result", states, /*verbose=*/VLOG_IS_ON(5));

  size_t valid_cnt = 0;
  for (size_t i = 0; i < states.size(); ++i) {
    std::vector<ir::Expr> best_exprs = states[i]->ir_schedule.GetModule().GetExprs();
    CHECK_EQ(best_exprs.size(), task_->lowered_funcs.size())
        << "RuntimeError: Expr size is not equal to LoweredFunc size in TaskOptimizer";
    auto init_funcs = optim::IRCopy(task_->lowered_funcs);
    std::vector<ir::LoweredFunc> valid_funcs;
    for (size_t j = 0; j < best_exprs.size(); ++j) {
      auto updated_f = UpdateFuncWithNewBody(task_->target, init_funcs[j], best_exprs[j]);
      if (PruneInvalid(updated_f)) {
        VLOG(4) << "PruneInvalid states-" << i;
        break;
      }
      valid_funcs.emplace_back(updated_f);
    }

    // all functions are validated, collect this state to be measured
    if (valid_funcs.size() == init_funcs.size()) {
      states[valid_cnt++] = states[i];
      measure_candidates->emplace_back(MeasureInput());
      measure_candidates->back().task          = task_;
      measure_candidates->back().lowered_funcs = std::move(valid_funcs);
    }
  }

  states.erase(states.begin() + valid_cnt, states.end());
  CHECK_EQ(states.size(), measure_candidates->size()) << "result size of states not equal to measure_candidates";
  VLOG(4) << "EvolutionarySearch return size=" << states.size() << ", valid count=" << valid_cnt;
  VLOG(4) << JoinStatesDebugString("TaskOptimizer::SearchOneRound-Result", states, /*verbose=*/VLOG_IS_ON(5));
  return states;
}

// detect the limit of avaliable shared memory on the currnet NVGPU with CUDA runtime
size_t GetGPUSharedMemoryLimit() {
#ifdef CINN_WITH_CUDA
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, device_id));
  VLOG(4) << utils::StringFormat("GPU-%d GPUSharedMemoryLimit=%d", device_id, prop.sharedMemPerBlock);
  return prop.sharedMemPerBlock;
#else
  return 0;
#endif
}

// detect the limit of avaliable local/stack memory on the currnet NVGPU with CUDA runtime
size_t GetGPULocalStackLimit() {
#ifdef CINN_WITH_CUDA
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, device_id));
  size_t limit = prop.totalGlobalMem / prop.multiProcessorCount / prop.maxThreadsPerMultiProcessor;
  VLOG(4) << utils::StringFormat(
      "GPU-%d totalGlobalMem=%lu,maxThreadsPerMultiProcessor=%d,multiProcessorCount=%d, calculated "
      "GPULocalStackLimit=%lu",
      device_id,
      prop.totalGlobalMem,
      prop.multiProcessorCount,
      prop.maxThreadsPerMultiProcessor,
      limit);
  return limit;
#else
  return 0;
#endif
}

// check whether usage of the specific memory type in the lowered_func exceeds hardware limit
bool IsGPUMemoryUsageExceedLimit(const ir::LoweredFunc& lowered_func,
                                 const ir::MemoryType& used_memory_type,
                                 const size_t limit_bytes) {
  std::unordered_set<std::string> visited;
  size_t used_bytes_cnt = 0;
  for (auto&& buf : lowered_func->temp_bufs) {
    VLOG(5) << "temp buf name=" << buf->name << ", numel=" << buf->numel() << ",dtype=" << buf->dtype;
    if (buf->memory_type == used_memory_type && !visited.count(buf->name)) {
      used_bytes_cnt += buf->numel() * buf->dtype.bytes();
      visited.insert(buf->name);
    }
  }
  VLOG(5) << "total used_bytes_cnt=" << used_bytes_cnt;
  return used_bytes_cnt >= limit_bytes;
}

bool TaskOptimizer::PruneInvalid(const ir::LoweredFunc& lowered_func) {
  static const size_t kGPUSharedMemoryLimitBytes = GetGPUSharedMemoryLimit();
  static const size_t kGPULocalStackLimitBytes   = GetGPULocalStackLimit();

  if (task_->target == common::DefaultNVGPUTarget()) {
    if (IsGPUMemoryUsageExceedLimit(lowered_func, ir::MemoryType::GPUShared, kGPUSharedMemoryLimitBytes)) {
      VLOG(5) << ir::MemoryType::GPUShared << " memory usage exceeds limit, func:\n" << lowered_func;
      return true;
    }

    if (IsGPUMemoryUsageExceedLimit(lowered_func, ir::MemoryType::GPULocal, kGPULocalStackLimitBytes)) {
      VLOG(5) << ir::MemoryType::GPULocal << " memory usage exceeds limit, func:\n" << lowered_func;
      return true;
    }
  }
  return false;
}

}  // namespace auto_schedule
}  // namespace cinn

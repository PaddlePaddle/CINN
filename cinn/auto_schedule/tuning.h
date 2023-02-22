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
#include <vector>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/ir/lowered_func.h"

namespace cinn {
namespace auto_schedule {

// alias a LoweredFunc array as FunctionGroup
using FunctionGroup = std::vector<ir::LoweredFunc>;
// alias a Graph::Group array as SubGraph
using SubGraphPtr = std::shared_ptr<hlir::framework::Graph::Group>;

// Options for tuning process
struct TuningOptions {
  // The number of tuning rounds, each round will tune several tasks,
  // each task involves TuningOptions.num_measure_trials measurements.
  int num_tuning_rounds = 1;

  // The number of measurement trials in a task, if it is 0,
  // that means the tunner will return the best
  // candidate of schedule config without measurement.
  int num_measure_trials = 10;

  // Every round TaskSchedule chooses some TuneTask(s) to optimize and run
  // several iterations of search algorithm for a task to generate samples.
  // Each iteration has num_samples_per_iteration samples.
  //
  // 1. if TuningOptions.num_measure_trials is 0, the autotune doesn't involve
  // hardware measurements. It predicts performance by cost model.
  //
  // 2. num_measure_trials % num_samples_per_iteration must equal 0.
  // In each round, autotune will run iterations until number of iterations
  // * num_samples_per_iteration equals num_measure_trials.
  int num_samples_per_iteration = 10;

  //////////////////////////////////////
  // Evolutionary Search Related Options
  //////////////////////////////////////

  // The number of picks from the stored database in each iteration
  // These are best performance recorded from previous generations
  //
  // Note the number doesn't guaranteed returns those topk when the
  // database doesn't have enough data. Evolutionary Search would get
  // as many as possible without throwing errors or warnings.
  int evolution_pick_database_topk = 8;

  // The number of initial populations at each generation. It contains
  // the picks from  database plus random generated samples.
  int evolution_init_population_num = 10;

  // The number of samples generated by cross over
  int evolution_cross_over_num = 10;

  // The fraction of random samples in num_samples_per_iteration.
  // So the num_samples_per_iteration would have (1 - eps_greedy) best
  // samples from evolutionary search and eps_greedy random samples.
  //
  // It explores the cases evolutionary search won't predict precisely
  float evolution_eps_greedy = 0.1f;
};

// Result of the tuning process
struct TuningResult {
  // Result of graph tuning
  std::vector<SubGraphPtr> subgraphs;
  // Result of schedule tuning
  std::vector<FunctionGroup> function_groups;
};

}  // namespace auto_schedule
}  // namespace cinn

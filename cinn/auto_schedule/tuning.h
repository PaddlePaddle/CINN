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

#include <vector>

#include "cinn/hlir/framework/node.h"
#include "cinn/ir/lowered_func.h"

namespace cinn {
namespace auto_schedule {

// Options for tuning process
struct TuningOptions {
  // The number of tuning rounds, each round will
  // involve TuningOptions.num_measure_trials measurements.
  int num_tuning_rounds = 1;
  // The number of measurement trials, if it is 0,
  // that means the tunner will return the best
  // candidate of schedule config without measurement.
  int num_measure_trials = 0;
};

// Result of the tuning process
struct TuningResult {
  // Result of graph tuning
  struct TunedGraph {
    std::vector<std::vector<hlir::framework::Node*>> groups;
  };

  // Result of schedule tuning in CINN IR
  struct OptimizedLoweredFuncs {
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  };

  std::vector<TunedGraph> tuned_graphs;
  std::vector<OptimizedLoweredFuncs> optimized_lowered;
};

}  // namespace auto_schedule
}  // namespace cinn

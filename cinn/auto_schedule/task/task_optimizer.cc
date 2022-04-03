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

#include "cinn/auto_schedule/search_strategy/evolutionary_search.h"

namespace cinn {
namespace auto_schedule {

TuningResult::OptimizedLoweredFuncs TaskOptimizer::Optimize(const TuningOptions& options) {
  // TODO(zhhsplendid): develop other optimize methods and configure the method by options.
  return OptimizeByEvolution(options);
}

TuningResult::OptimizedLoweredFuncs TaskOptimizer::OptimizeByEvolution(const TuningOptions& options) {
  // TODO(zhhsplendid): finish this function
  return TuningResult::OptimizedLoweredFuncs();
}

}  // namespace auto_schedule
}  // namespace cinn

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

#include "cinn/auto_schedule/measure/schedule_measurer.h"

namespace cinn {
namespace auto_schedule {

ScheduleMeasurer::ScheduleMeasurer(ScheduleBuilder* builder, ScheduleRunner* runner)
    : builder_(builder), runner_(runner) {}

std::vector<MeasureResult> ScheduleMeasurer::Measure(const std::vector<MeasureInput>& inputs) {
  std::vector<MeasureResult> results;
  for (auto i = 0; i < inputs.size(); ++i) {
    auto m_start = std::chrono::steady_clock::now();
    auto&& input = inputs.at(i);

    BuildResult build_res = builder_->Build(input);
    MeasureResult res     = runner_->Run(input, build_res);

    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start);
    // use the time span counted in measurer
    res.elapsed_time = static_cast<double>(time_span.count());
    VLOG(5) << "Measurement-" << i << " cost " << res.elapsed_time << "us";
    results.emplace_back(std::move(res));
  }

  VLOG(4) << "Measure " << inputs.size() << " tests";
  return results;
}

}  // namespace auto_schedule
}  // namespace cinn

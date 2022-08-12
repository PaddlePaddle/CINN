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

#include <exception>

namespace cinn {
namespace auto_schedule {

ScheduleMeasurer::ScheduleMeasurer(ScheduleBuilder* builder, ScheduleRunner* runner, int num_threads)
    : builder_(builder), runner_(runner), num_threads_(num_threads) {}

std::vector<MeasureResult> ScheduleMeasurer::Measure(const std::vector<MeasureInput>& inputs) {
  if (inputs.empty()) {
    LOG(WARNING) << "inputs is empty";
    return {};
  }
  std::vector<BuildResult> build_results(inputs.size());
  std::vector<MeasureResult> results(inputs.size());

  auto build_fn = [builder = builder_, &inputs, &build_results, &results](int index) {
    VLOG(6) << "Build candidate:" << index;
    auto m_start = std::chrono::steady_clock::now();
    try {
      build_results[index] = builder->Build(inputs[index]);
    } catch (std::exception& e) {
      results[index].error_msg = utils::StringFormat("Build failed, error:%s\n", e.what());
    }
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start);
    results[index].elapsed_time += static_cast<double>(time_span.count());
  };

  auto run_fn = [runner = runner_, &inputs, &build_results, &results](int index) {
    VLOG(6) << "Run candidate:" << index;
    auto m_start = std::chrono::steady_clock::now();
    try {
      results[index] = runner->Run(inputs[index], build_results[index]);
    } catch (std::exception& e) {
      results[index].error_msg = utils::StringFormat("Run failed, error:%s\n", e.what());
    }
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start);
    results[index].elapsed_time += static_cast<double>(time_span.count());
  };

  if (num_threads_ > 1) {
    // TODO
  } else {
    for (int i = 0; i < inputs.size(); ++i) {
      build_fn(i);
      run_fn(i);
    }
  }

  VLOG(4) << "Measure " << inputs.size() << " candidates";
  return results;
}

}  // namespace auto_schedule
}  // namespace cinn

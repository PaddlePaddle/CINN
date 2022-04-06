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

#include <gtest/gtest.h>

#include <memory>

#include "cinn/auto_schedule/measure/schedule_measurer.h"
#include "cinn/auto_schedule/measure/simple_builder.h"
#include "cinn/auto_schedule/measure/simple_runner.h"

namespace cinn {
namespace auto_schedule {

TEST(ScheduleMeasurer, Basic) {
  auto builder  = std::make_unique<SimpleBuilder>(nullptr);
  auto runner   = std::make_unique<SimpleRunner>(1);
  auto measurer = std::make_unique<ScheduleMeasurer>(builder.get(), runner.get());
  std::vector<MeasureInput> inputs(2);
  std::vector<MeasureResult> results = measurer->Measure(inputs);
  ASSERT_EQ(2, results.size());
}

}  // namespace auto_schedule
}  // namespace cinn

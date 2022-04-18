// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>

#include <string>
#include <vector>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/pe/schedule_param.pb.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/lang/compute.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace pe {

void NewScheduleInjectiveCPU(ir::IRSchedule &ir_sch,
                             const std::vector<int> &output_shape,
                             const common::Target &target,
                             bool vectorizable = true);

void NewCudaScheduleInjective(ir::IRSchedule &ir_sch,
                              const std::vector<int> &output_shape,
                              const common::Target &target);

void NewCudaScheduleMul(ir::IRSchedule &ir_sch, const std::vector<int> &output_shape, const common::Target &target);
}  // namespace pe
}  // namespace hlir
}  // namespace cinn

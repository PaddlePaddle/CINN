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

#include "cinn/auto_schedule/task/tune_task.h"

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>

#include <string>
#include <vector>

#include "cinn/common/cinn_value.h"
#include "cinn/common/graph_utils.h"
#include "cinn/common/shared.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/lang/placeholder.h"

namespace cinn {
namespace auto_schedule {}  // namespace auto_schedule
}  // namespace cinn

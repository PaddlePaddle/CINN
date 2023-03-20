// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/frontend/optimize.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/lowered_func.h"

namespace cinn {
namespace tests {

std::shared_ptr<hlir::framework::Graph> OptimizeByPass(frontend::Program& program, const common::Target& target);

std::vector<ir::LoweredFunc> LowerFusionGroup(std::shared_ptr<hlir::framework::Graph> graph,
                                              std::shared_ptr<hlir::framework::Graph::Group> group,
                                              const common::Target& target,
                                              bool apply_manual_schedule = true);

ir::IRSchedule MakeIRSchedule(const std::vector<ir::LoweredFunc>& lowered_funcs);

std::vector<ir::LoweredFunc> OptimizeBySchedule(const ir::IRSchedule& schedule,
                                                const std::vector<ir::LoweredFunc>& original_funcs,
                                                const common::Target& target);

ir::Module BuildIRModule(const std::vector<ir::LoweredFunc>& lowered_funcs, const common::Target& target);

std::string GenSourceCode(const ir::Module& ir_module, const common::Target& target);

std::vector<hlir::framework::Instruction> BuildExecution(const ir::Module& ir_module,
                                                         const common::Target& target,
                                                         hlir::framework::Scope* scope);

}  // namespace tests
}  // namespace cinn

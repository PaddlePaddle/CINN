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

#include <string>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/lowered_func.h"

namespace cinn {
namespace auto_schedule {

/**
 * A class containing context information for tuning-task. The difference
 * between this class and TuneTask is that the data in this context is only
 * needed by autotune while the TuneTask contains some information for whole
 * compiler, such as Graph, GraphCompiler.
 */
class TuneContext {
 public:
  std::vector<ir::LoweredFunc> lowered_funcs;
  common::Target target;

  std::vector<ir::Expr> GetLoweredFuncBodyExprs() const;

  void SetLoweredFuncBodyExprs(const std::vector<ir::Expr>& exprs);
};

}  // namespace auto_schedule
}  // namespace cinn

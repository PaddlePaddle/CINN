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

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/ir_base.h"

namespace cinn {
namespace utils {

std::vector<ir::Expr> FusedGraphToExpr(const std::vector<std::vector<hlir::framework::Node*>>& graph);

std::vector<ir::Expr> NodeToExpr(const hlir::framework::Node& node);

std::vector<ir::Expr> FusedNodeGroupToExpr(const std::vector<hlir::framework::Node*>& node_group);

}  // namespace utils
}  // namespace cinn

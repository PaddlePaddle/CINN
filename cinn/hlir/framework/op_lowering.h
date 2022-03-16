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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/packed_func.h"

// Fusion Op lowering, there are four kinds of lowering function:
// Elementwise/Broadcast/Injective、Reduce、OutEWiseFusable、Opaque.
// Elementwise/Broadcast/Injective Ops is with same shcedule.
// Reduce、OutEWiseFusable、Opaque are using different schedule.

namespace cinn {
namespace hlir {
namespace framework {

using Group  = std::shared_ptr<Graph::Group>;
using Groups = std::vector<Group>;
using common::Target;

class OpLoweringHelper {
 public:
  OpLoweringHelper(const absl::flat_hash_map<std::string, Type>&,
                   const absl::flat_hash_map<std::string, shape_t>&,
                   const Target&);
  ir::LoweredFunc Lowering(const Group& group);

 private:
  ir::LoweredFunc ElementwiseOpLowering(const Group& group);
  ir::LoweredFunc ReduceOpLowering(const Group& group);
  ir::LoweredFunc FusableOpLowering(const Group& group);
  ir::LoweredFunc OpaqueOpLowering(const Group& group);

  Target target_;
  const absl::flat_hash_map<std::string, Type>& type_dict_;
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

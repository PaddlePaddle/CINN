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
#include <absl/container/flat_hash_map.h>

#include <map>
#include <string>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/schedule_desc.pb.h"
#include "cinn/utils/registry.h"
#include "cinn/utils/type_defs.h"

namespace cinn {
namespace ir {

// A ScheduleDesc describe the scheduling process of an ir::ModuleExpr, it records
// all transform/getting operations executed by a corresponding ir::IRSchedule.
// A ScheduleDesc can be serialized to JSON format and saved to file. For deserializing,
// it can be re-applied to a new IRSchedule that is initialzied by a semantics-euqal
// original ir::ModuleExpr, and then achieves the same result.

class IRSchedule;  // forward declartion to avoid cross-reference
class ScheduleDesc {
 public:
  // each operation executed through IRSchedule is recorded as a step
  struct Step {
    std::string type;  // step name
    absl::flat_hash_map<std::string, std::vector<Expr>> inputs;
    utils::AttributeMap attrs;
    std::vector<Expr> outputs;
    Step() = default;
    Step(std::string type_i,
         absl::flat_hash_map<std::string, std::vector<Expr>> inputs_i,
         utils::AttributeMap attrs_i,
         std::vector<Expr> outputs_i)
        : type(type_i), inputs(inputs_i), attrs(attrs_i), outputs(outputs_i) {}
  };
  std::vector<Step> steps;  // all operations are recorded in order.

  // Re-applied a scheduling process represented as a proto::ScheduleDesc to a new IRSchedule object
  static std::vector<Expr> ReplayWithProto(const proto::ScheduleDesc& desc_proto, IRSchedule* sch);

  // Append a new step
  void Append(Step&& step);

  // Pop the last step
  void Pop();

  // Replay this description to a new IRSchedule that is initialzied
  // by a semantics-euqal original ModuleExpr
  void Replay(IRSchedule* schedule) const;

  // convert to a proto::ScheduleDesc object
  proto::ScheduleDesc ToProto() const;
};

}  // namespace ir
}  // namespace cinn

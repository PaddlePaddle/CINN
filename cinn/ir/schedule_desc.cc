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

#include "cinn/ir/schedule_desc.h"

#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace ir {

class PackedStepContext {
 public:
  PackedStepContext(const ScheduleDesc::Step& step_desc, const ExprNameMap& exprs);
  Expr Input(size_t idx);
  std::vector<Expr> Inputs(size_t idx);
  Expr* Output(size_t idx);
  std::vector<Expr*> Outputs(size_t idx);
  template <typename DataType>
  DataType Attr(size_t idx) const;

 private:
  const ScheduleDesc::Step& step_desc_;
  const ExprNameMap& exprs_;
};

// implement context

// ---- register StepKind
// 1. define marco: uniform apply function

// ---- ScheduleDesc
ScheduleDesc::ScheduleDesc(const proto::ScheduleDesc& desc_proto) {}

void ScheduleDesc::Append(Step&& step) {}

void ScheduleDesc::Replay(IRSchedule* schedule) {}

proto::ScheduleDesc ScheduleDesc::ToProto() const {}

}  // namespace ir
}  // namespace cinn

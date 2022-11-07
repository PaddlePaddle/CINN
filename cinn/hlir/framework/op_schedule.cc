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

#include "cinn/hlir/framework/op_schedule.h"

#include "cinn/common/cinn_value.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_schedule.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace framework {

CINNSchedule GetInjectiveScheduleFunc(const std::vector<std::vector<int>>& output_shapes,
                                      const Target& target,
                                      bool vectorizable) {
  return CINNSchedule([=](lang::Args args, lang::RetValue* ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty()) << "The input argument of InjectiveSchedule is empty! Please check.\n";
      common::CINNValuePack arg_pack = args[0];
      std::vector<Expr> vec_ast;
      for (int i = 0; i < arg_pack.size(); i++) {
        if (arg_pack[i].is_expr()) {
          Expr temp = arg_pack[i];
          vec_ast.emplace_back(temp);
        }
      }
      CHECK(!vec_ast.empty());
      ir::ModuleExpr mod_expr(vec_ast);
      ir::IRSchedule ir_sch(mod_expr);
      ir_sch.MergeExprs();
      pe::IRElementwiseSchedule(ir_sch, output_shapes.front(), target);
      /*
      if (target.arch == Target::Arch::NVGPU) {
        pe::IRCudaScheduleInjective(ir_sch, output_shapes.front(), target);
      } else if (target.arch == Target::Arch::X86) {
        pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target, vectorizable);
      }
      */
      std::vector<common::CINNValue> res{common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
      *ret = common::CINNValuePack{res};
    } else {
      CHECK(!args.empty()) << "The input argument of InjectiveSchedule is empty! Please check.\n";
      common::CINNValuePack arg_pack = args[0];
      Expr out                       = arg_pack[0];
      poly::StageMap stages          = arg_pack[1];
      CHECK(out.as_tensor());
      CHECK_EQ(arg_pack.size(), 2UL);
      if (target.arch == Target::Arch::NVGPU) {
        pe::CudaScheduleInjective(stages[out.as_tensor_ref()], output_shapes.front(), target);
      } else if (target.arch == Target::Arch::X86) {
        pe::ScheduleInjectiveCPU(stages[out.as_tensor_ref()], output_shapes.front(), target, vectorizable);
      }
      *ret = arg_pack;
    }
  });
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

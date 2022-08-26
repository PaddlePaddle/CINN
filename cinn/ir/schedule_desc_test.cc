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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/lower.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/utils/string.h"
#include "cinn/utils/type_defs.h"

namespace cinn {
namespace ir {

TEST(ScheduleDesc, Append) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});

  auto func     = cinn::lang::LowerVec("test_split_and_fuse1", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  LOG(INFO) << "Initial IR:" << ast_expr;
  ir::IRSchedule replay_sch;
  replay_sch.SetExprs({optim::IRCopy(ast_expr)});
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ScheduleDesc desc;
  ir::IRSchedule ir_sch(mod_expr);
  auto fused = ir_sch.Fuse("B", {0, 1});
  desc.Append(ScheduleDesc::Step(
      "FuseWithBlockName", {}, {{"block_name", std::string("B")}, {"loops_index", std::vector<int>({0, 1})}}, {fused}));
  auto splited = ir_sch.Split(fused, {4, -1});
  desc.Append(ScheduleDesc::Step(
      "Split", {{"loop", std::vector<Expr>({fused})}}, {{"factors", std::vector<int>({4, -1})}}, splited));
  LOG(INFO) << "After split {4, -1}, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  auto loops = ir_sch.GetLoops("B");
  desc.Append(ScheduleDesc::Step("GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  fused = ir_sch.Fuse(loops);
  desc.Append(ScheduleDesc::Step("Fuse", {{"loops", loops}}, {}, {fused}));
  splited = ir_sch.Split(fused, {256, -1});
  desc.Append(ScheduleDesc::Step(
      "Split", {{"loop", std::vector<Expr>({fused})}}, {{"factors", std::vector<int>({256, -1})}}, splited));
  LOG(INFO) << "After split {256, -1}, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  // LOG(INFO) << "split_and_fuse1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_split_and_fuse1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i_j_fused_0_i_j_fused_1_fused_0 = 0; i_j_fused_0_i_j_fused_1_fused_0 < 256; i_j_fused_0_i_j_fused_1_fused_0 += 1) {
    for (int32_t i_j_fused_0_i_j_fused_1_fused_1 = 0; i_j_fused_0_i_j_fused_1_fused_1 < 4; i_j_fused_0_i_j_fused_1_fused_1 += 1) {
      B[((4 * i_j_fused_0_i_j_fused_1_fused_0) + i_j_fused_0_i_j_fused_1_fused_1)] = A[((4 * i_j_fused_0_i_j_fused_1_fused_0) + i_j_fused_0_i_j_fused_1_fused_1)];
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
  desc.Replay(&replay_sch);
  LOG(INFO) << "After Replay, IR is : " << replay_sch.GetModule().GetExprs().at(0);
  ASSERT_EQ(utils::GetStreamCnt(ir_sch.GetModule().GetExprs().at(0)),
            utils::GetStreamCnt(replay_sch.GetModule().GetExprs().at(0)));
}

}  // namespace ir
}  // namespace cinn

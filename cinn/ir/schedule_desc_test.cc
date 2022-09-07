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

std::vector<ir::LoweredFunc> ElementwiseCopyExpr(const std::vector<int>& shape,
                                                 const Target& target,
                                                 const std::string& func_name) {
  CHECK_EQ(shape.size(), 2) << "size of shape shoule be 2";
  Expr M(shape[0]);
  Expr N(shape[1]);

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  return cinn::lang::LowerVec(func_name, stages, {A, B}, {}, {}, nullptr, target, true);
}

class TestScheduleDesc : public ::testing::Test {
 public:
  void SetUp() override { Context::Global().ResetNameId(); }
  // void TearDown() override {}

 private:
  IRSchedule MakeIRSchedule(const std::vector<ir::LoweredFunc>& lowered_funcs) {
    std::vector<Expr> exprs;
    for (auto&& func : lowered_funcs) {
      exprs.emplace_back(optim::IRCopy(func->body));
    }
    return ir::IRSchedule(ir::ModuleExpr(exprs));
  }

  void CheckExpr(const ModuleExpr& lfs, const ModuleExpr& rhs) {
    lfs_exprs = lfs.GetExprs();
    rfs_exprs = rfs.GetExprs();
    ASSERT_EQ(lfs_exprs.size(), rfs_exprs.size());
    for (auto i = 0; i < lfs_exprs.size(); ++i) {
      ASSERT_EQ(utils::GetStreamCnt(lfs_exprs.at(i)), utils::GetStreamCnt(rfs_exprs.at(i)));
    }
  }

  std::string SourceCodeGen(const ModuleExpr& module_expr,
                            const Target& target,
                            std::vector<ir::LoweredFunc>& lowered_funcs) {
    auto exprs = module_expr.GetExprs();
    ASSERT_EQ(exprs.size(), lowered_funcs.size());
    Module::Builder builder("test_module", target);
    for (auto i = 0; i < lowered_funcs.size(); ++i) {
      auto&& func = lowered_funcs.at(i);
      func->body  = exprs.at(i);
      builder.AddFunction(func);
    }
    auto module = builder.Build();
    CodeGenC codegen(target);
    codegen.SetInlineBuiltinCodes(false);
    std::string source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  }
};

TEST_F(TestScheduleDesc, Append) {
  Target target = common::DefaultHostTarget();
  auto funcs    = ElementwiseCopyExpr({32, 32}, target, "test_split_and_fuse1");
  VLOG(3) << "Initial IR:" << ast_expr;

  ir::IRSchedule ir_sch     = MakeIRSchedule(funcs);
  ir::IRSchedule replay_sch = MakeIRSchedule(funcs);
  ScheduleDesc desc;

  auto fused = ir_sch.Fuse("B", {0, 1});
  desc.Append(ScheduleDesc::Step(
      "FuseWithBlockName", {}, {{"block_name", std::string("B")}, {"loops_index", std::vector<int>({0, 1})}}, {fused}));
  auto splited = ir_sch.Split(fused, {4, -1});
  desc.Append(ScheduleDesc::Step(
      "Split", {{"loop", std::vector<Expr>({fused})}}, {{"factors", std::vector<int>({4, -1})}}, splited));

  auto loops = ir_sch.GetLoops("B");
  desc.Append(ScheduleDesc::Step("GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  fused = ir_sch.Fuse(loops);
  desc.Append(ScheduleDesc::Step("Fuse", {{"loops", loops}}, {}, {fused}));
  splited = ir_sch.Split(fused, {256, -1});
  desc.Append(ScheduleDesc::Step(
      "Split", {{"loop", std::vector<Expr>({fused})}}, {{"factors", std::vector<int>({256, -1})}}, splited));

  desc.Replay(&replay_sch);
  VLOG(3) << "Scheduled IR:" << ir_sch.GetModule().GetExprs().at(0)
          << "\nReplay IR: " << replay_sch.GetModule().GetExprs().at(0);

  CheckExpr(ir_sch.GetModule(), replay_sch.GetModule());
  ASSERT_EQ(utils::Trim(SourceCodeGen(ir_sch.GetModule(), target, funcs)),
            utils::Trim(SourceCodeGen(replay_sch.GetModule(), target, funcs)));
}

}  // namespace ir
}  // namespace cinn

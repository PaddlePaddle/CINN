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

#include "cinn/ir/ir_schedule.h"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <tuple>
#include <vector>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/codegen_c_x86.h"
#include "cinn/cinn.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/lower.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/remove_schedule_block.h"

namespace cinn {
namespace backends {

TEST(IrSchedule, split_and_fuse1) {
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
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops();

  auto fused   = ir_sch.Fuse(loops);
  auto splited = ir_sch.Split(fused, {4, -1});
  VLOG(3) << "After split {4, -1}, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  loops   = ir_sch.GetLoops();
  fused   = ir_sch.Fuse(loops);
  splited = ir_sch.Split(fused, {256, -1});
  VLOG(3) << "After split {256, -1}, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  func[0]->body = ir_sch.GetModule().GetExprs().at(0);

  optim::RemoveScheduleBlock(&func[0]->body);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "split_and_fuse1 source code is :\n" << source_code;

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
}

TEST(IrSchedule, split_and_fuse2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});

  auto func     = cinn::lang::LowerVec("test_split_and_fuse2", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops();

  auto fused   = ir_sch.Fuse(loops);
  auto splited = ir_sch.Split(fused, {-1, 20});
  VLOG(3) << "After split {-1, 20}, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  func[0]->body = ir_sch.GetModule().GetExprs().at(0);

  optim::RemoveScheduleBlock(&func[0]->body);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "split_and_fuse2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_split_and_fuse2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i_j_fused_0 = 0; i_j_fused_0 < 52; i_j_fused_0 += 1) {
    for (int32_t i_j_fused_1 = 0; i_j_fused_1 < 20; i_j_fused_1 += 1) {
      if ((((20 * i_j_fused_0) + i_j_fused_1) < 1024)) {
        B[((20 * i_j_fused_0) + i_j_fused_1)] = A[((20 * i_j_fused_0) + i_j_fused_1)];
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, reorder1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");

  auto stages = CreateStages({A, B});

  auto func     = cinn::lang::LowerVec("test_reorder1", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops();

  auto splited = ir_sch.Split(0, {-1, 4});
  splited      = ir_sch.Split(2, {-1, 2});

  ir_sch.Reorder({4, 0});

  func[0]->body = ir_sch.GetModule().GetExprs().at(0);

  optim::RemoveScheduleBlock(&func[0]->body);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "reorder1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_reorder1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t k = 0; k < 32; k += 1) {
    for (int32_t i_1 = 0; i_1 < 4; i_1 += 1) {
      for (int32_t j_0 = 0; j_0 < 16; j_0 += 1) {
        for (int32_t j_1 = 0; j_1 < 2; j_1 += 1) {
          for (int32_t i_0 = 0; i_0 < 8; i_0 += 1) {
            B[((4096 * i_0) + ((1024 * i_1) + ((64 * j_0) + ((32 * j_1) + k))))] = A[((4096 * i_0) + ((1024 * i_1) + ((64 * j_0) + ((32 * j_1) + k))))];
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, reorder2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");

  auto stages = CreateStages({A, B});

  auto func     = cinn::lang::LowerVec("test_reorder2", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops();

  auto splited = ir_sch.Split(0, {-1, 4});
  splited      = ir_sch.Split(2, {-1, 2});

  ir_sch.Reorder({4, 2, 3, 1, 0});

  func[0]->body = ir_sch.GetModule().GetExprs().at(0);

  optim::RemoveScheduleBlock(&func[0]->body);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "reorder2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_reorder2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t k = 0; k < 32; k += 1) {
    for (int32_t j_0 = 0; j_0 < 16; j_0 += 1) {
      for (int32_t j_1 = 0; j_1 < 2; j_1 += 1) {
        for (int32_t i_1 = 0; i_1 < 4; i_1 += 1) {
          for (int32_t i_0 = 0; i_0 < 8; i_0 += 1) {
            B[((4096 * i_0) + ((1024 * i_1) + ((64 * j_0) + ((32 * j_1) + k))))] = A[((4096 * i_0) + ((1024 * i_1) + ((64 * j_0) + ((32 * j_1) + k))))];
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, reorder3) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");

  auto stages = CreateStages({A, B});

  auto func     = cinn::lang::LowerVec("test_reorder3", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops();

  auto splited = ir_sch.Split(0, {-1, 5});
  splited      = ir_sch.Split(2, {-1, 2});

  ir_sch.Reorder({3, 1, 2, 0, 4});

  func[0]->body = ir_sch.GetModule().GetExprs().at(0);

  optim::RemoveScheduleBlock(&func[0]->body);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "reorder3 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_reorder3(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t j_1 = 0; j_1 < 2; j_1 += 1) {
    for (int32_t i_1 = 0; i_1 < 5; i_1 += 1) {
      for (int32_t j_0 = 0; j_0 < 16; j_0 += 1) {
        for (int32_t i_0 = 0; i_0 < 7; i_0 += 1) {
          if ((((5 * i_0) + i_1) < 32)) {
            for (int32_t k = 0; k < 32; k += 1) {
              B[((5120 * i_0) + ((1024 * i_1) + ((64 * j_0) + ((32 * j_1) + k))))] = A[((5120 * i_0) + ((1024 * i_1) + ((64 * j_0) + ((32 * j_1) + k))))];
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, reorder4) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");

  auto stages = CreateStages({A, B});

  auto func     = cinn::lang::LowerVec("test_reorder4", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops();

  auto splited = ir_sch.Split(0, {-1, 10});
  splited      = ir_sch.Split(2, {-1, 5});

  ir_sch.Reorder({0, 2, 1, 3, 4});

  func[0]->body = ir_sch.GetModule().GetExprs().at(0);

  optim::RemoveScheduleBlock(&func[0]->body);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "reorder4 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_reorder4(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
    for (int32_t j_0 = 0; j_0 < 7; j_0 += 1) {
      for (int32_t i_1 = 0; i_1 < 10; i_1 += 1) {
        if ((((10 * i_0) + i_1) < 32)) {
          for (int32_t j_1 = 0; j_1 < 5; j_1 += 1) {
            if ((((5 * j_0) + j_1) < 32)) {
              for (int32_t k = 0; k < 32; k += 1) {
                B[((10240 * i_0) + ((1024 * i_1) + ((160 * j_0) + ((32 * j_1) + k))))] = A[((10240 * i_0) + ((1024 * i_1) + ((160 * j_0) + ((32 * j_1) + k))))];
              };
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

}  // namespace backends
}  // namespace cinn

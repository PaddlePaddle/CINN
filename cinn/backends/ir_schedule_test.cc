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
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/cinn.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/lower.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/remove_schedule_block.h"
#include "cinn/optim/unroll_loops.h"
#include "cinn/optim/vectorize_loops.h"

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
  auto fused   = ir_sch.Fuse("B", {0, 1});
  auto splited = ir_sch.Split(fused, {4, -1});
  LOG(INFO) << "After split {4, -1}, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  auto loops = ir_sch.GetLoops("B");
  fused      = ir_sch.Fuse(loops);
  splited    = ir_sch.Split(fused, {256, -1});
  LOG(INFO) << "After split {256, -1}, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "split_and_fuse1 source code is :\n" << source_code;

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
  auto loops = ir_sch.GetLoops("B");

  auto fused   = ir_sch.Fuse(loops);
  auto splited = ir_sch.Split(fused, {-1, 20});
  VLOG(3) << "After split {-1, 20}, IR is : " << ir_sch.GetModule().GetExprs().at(0);

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

  auto splited = ir_sch.Split("B", 0, {-1, 4});
  splited      = ir_sch.Split("B", 2, {-1, 2});

  auto loops = ir_sch.GetLoops("B");
  ir_sch.Reorder({loops[4], loops[0]});

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

  auto splited = ir_sch.Split("B", 0, {-1, 4});
  splited      = ir_sch.Split("B", 2, {-1, 2});

  ir_sch.Reorder("B", {4, 2, 3, 1, 0});

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
  auto all_blocks = ir_sch.GetAllBlocks();
  auto loops      = ir_sch.GetLoops(all_blocks[0]);

  auto splited = ir_sch.Split(loops[0], {-1, 5});
  splited      = ir_sch.Split("B", 2, {-1, 2});

  ir_sch.Reorder("B", {3, 1, 2, 0, 4});

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

  auto all_blocks = ir_sch.GetAllBlocks();
  auto block_b    = ir_sch.GetBlock("B");
  auto loops      = ir_sch.GetLoops(block_b);

  auto splited = ir_sch.Split("B", 0, {-1, 10});
  splited      = ir_sch.Split("B", 2, {-1, 5});

  ir_sch.Reorder("B", {0, 2, 1, 3, 4});

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

TEST(IrSchedule, parallel) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  auto func   = cinn::lang::LowerVec("test_parallel", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK(!loops.empty());
  ir_sch.Parallel(loops[0]);
  LOG(INFO) << "After parallel , IR is : \n" << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "parallel source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_parallel(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  int num_task = max_concurrency();
  omp_set_num_threads(num_task);
  auto flambda = [=](int task_id, int num_task) -> int {
    int n_per_task = (((32 + num_task) - 1) / num_task);
    for (int32_t i = (task_id * n_per_task); i < 32 && i < ((task_id + 1) * n_per_task); i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        B[((32 * i) + j)] = A[((32 * i) + j)];
      };
    }
    return 0;
  };
#pragma omp parallel num_threads(num_task)
  {
    int task_id = omp_get_thread_num();
    flambda(task_id, num_task);
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, vectorize) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  auto func   = cinn::lang::LowerVec("test_vectorize", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 2U);
  ir_sch.Vectorize(loops[1], 16);
  std::string origin = utils::GetStreamCnt(func[0]);
  LOG(INFO) << "After Vectorize , func is : \n" << func[0];
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_vectorize (_A, _B)
{
  ScheduleBlock(root)
  {
    for (i, 0, 32)
    {
      vectorize_16 for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
}
)ROC"));
  optim::VectorizeLoops(&func[0]->body, target);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "Vectorize source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_vectorize(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 2; j += 1) {
      B[StackVec<16,int32_t>::Ramp(((32 * i) + (16 * j)), 1, 16)] = StackedVec<float,16>::Load(A,((32 * i) + (16 * j)));
    };
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, unroll) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  auto func   = cinn::lang::LowerVec("test_unroll", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 2U);
  ir_sch.Unroll(loops[1]);
  std::string origin = utils::GetStreamCnt(func[0]);
  LOG(INFO) << "After unroll , func is : \n" << func[0];
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_unroll (_A, _B)
{
  ScheduleBlock(root)
  {
    for (i, 0, 32)
    {
      unroll for (j, 0, 2)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
}
)ROC"));
  optim::UnrollLoop(&func[0]->body);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "Unroll source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_unroll(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    B[(2 * i)] = A[(2 * i)];
    B[(2 * i)] = A[(2 * i)];
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, bind) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  auto func   = cinn::lang::LowerVec("test_bind", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 2U);
  ir_sch.Bind(loops[0], "blockIdx.x");
  std::string origin = utils::GetStreamCnt(func[0]);
  LOG(INFO) << "After bind , func is : \n" << func[0];
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_bind (_A, _B)
{
  ScheduleBlock(root)
  {
    thread_bind_blockIdx.x for (i, 0, 32)
    {
      for (j, 0, 2)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
}
)ROC"));
}

TEST(IrSchedule, compute_at1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(i, j, k); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec("test_compute_at1", stages, {A, C}, {}, {}, nullptr, target, true);
  LOG(INFO) << "func size is : " << func.size();
  for (auto& i : func) LOG(INFO) << i->body;

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops   = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[1]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "compute_at1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void test_compute_at1(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 32768 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t ax0 = 0; ax0 < 32; ax0 += 1) {
        B[((1024 * i) + ((32 * j) + ax0))] = A[((1024 * i) + ((32 * j) + ax0))];
      };
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = B[((1024 * i) + ((32 * j) + k))];
      };
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at2) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {M, M}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {N, N}, [&](Var i, Var j) { return B(i + j, i + j); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec("test_compute_at1", stages, {A, C}, {}, {}, nullptr, target, true);
  LOG(INFO) << "func size is : " << func.size();
  for (auto& i : func) LOG(INFO) << i->body;

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops   = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[0]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "compute_at2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void test_compute_at1(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 4096 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t ax0 = 0; ax0 < 32; ax0 += 1) {
      for (int32_t ax1 = 0; ax1 < 32; ax1 += 1) {
        B[((64 * ax0) + ((64 * i) + (ax1 + i)))] = A[((64 * ax0) + ((64 * i) + (ax1 + i)))];
      };
    };
    for (int32_t j = 0; j < 32; j += 1) {
      C[((32 * i) + j)] = B[((65 * i) + (65 * j))];
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at3) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {M, M}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {M, M}, [&](Var i, Var j) { return B(i, j); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec("test_compute_at3", stages, {A, C}, {}, {}, nullptr, target, true);
  LOG(INFO) << "func size is : " << func.size();
  for (auto& i : func) LOG(INFO) << i->body;

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");

  auto fused   = ir_sch.Fuse("C", {0, 1});
  auto splited = ir_sch.Split(fused, {32, -1});

  auto loops = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[1]);

  VLOG(1) << "After ComputeAt, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at3 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void test_compute_at3(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 4096 ];
  float* B = _B_temp_buffer;
  for (int32_t i_j_fused_0 = 0; i_j_fused_0 < 32; i_j_fused_0 += 1) {
    for (int32_t i_j_fused_1 = 0; i_j_fused_1 < 128; i_j_fused_1 += 1) {
      B[((64 * (i_j_fused_1 / 64)) + ((((128 * i_j_fused_0) + i_j_fused_1) & 63) + (128 * i_j_fused_0)))] = A[((64 * (i_j_fused_1 / 64)) + ((((128 * i_j_fused_0) + i_j_fused_1) & 63) + (128 * i_j_fused_0)))];
      C[((128 * i_j_fused_0) + i_j_fused_1)] = B[((128 * i_j_fused_0) + i_j_fused_1)];
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

}  // namespace backends
}  // namespace cinn

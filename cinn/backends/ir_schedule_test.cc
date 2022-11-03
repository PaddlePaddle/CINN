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
    serial for (i, 0, 32)
    {
      vectorize[16] for (j, 0, 32)
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
    serial for (i, 0, 32)
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
    B[(1 + (2 * i))] = A[(1 + (2 * i))];
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
    thread_bind[blockIdx.x] for (i, 0, 32)
    {
      serial for (j, 0, 2)
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
TEST(IrSchedule, simple_compute_at) {
  Context::Global().ResetNameId();
  Expr M(128);
  Expr N(10);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_simple_compute_at", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto fused   = ir_sch.Fuse("B", {0, 1});
  auto splited = ir_sch.Split(fused, {-1, 1024});

  fused        = ir_sch.Fuse("C", {0, 1});
  splited      = ir_sch.Split(fused, {-1, 1024});
  auto block_b = ir_sch.GetBlock("B");
  ir_sch.SimpleComputeAt(block_b, splited[1]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "simple_compute_at source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at0(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 128, 10 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i_j_fused_0 = 0; i_j_fused_0 < 2; i_j_fused_0 += 1) {
    for (int32_t i_j_fused_1 = 0; i_j_fused_1 < 1024; i_j_fused_1 += 1) {
      if ((((1024 * i_j_fused_0) + i_j_fused_1) < 1280)) {
      {
        B[((1024 * i_j_fused_0) + i_j_fused_1)] = A[((1024 * i_j_fused_0) + i_j_fused_1)];
        C[((1024 * i_j_fused_0) + i_j_fused_1)] = B[((1024 * i_j_fused_0) + i_j_fused_1)];
      }
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at0) {
  Context::Global().ResetNameId();
  Expr M(128);
  Expr N(10);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_compute_at0", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto fused   = ir_sch.Fuse("B", {0, 1});
  auto splited = ir_sch.Split(fused, {-1, 1024});

  fused        = ir_sch.Fuse("C", {0, 1});
  splited      = ir_sch.Split(fused, {-1, 1024});
  auto block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeAt(block_b, splited[1]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at0 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
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
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(i, j, k); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_compute_at1", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

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
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
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
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at2) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {M, M}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {N, N}, [&](Var i, Var j) { return B(i + j, i + j); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_compute_at2", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

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
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 64, 64 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
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
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at3) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {M, M}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {M, M}, [&](Var i, Var j) { return B(i, j); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_compute_at3", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");

  auto fused   = ir_sch.Fuse("C", {0, 1});
  auto splited = ir_sch.Split(fused, {32, -1});

  auto loops = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[0]);

  VLOG(1) << "After ComputeAt, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at3 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at3(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 64, 64 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i_j_fused_0 = 0; i_j_fused_0 < 32; i_j_fused_0 += 1) {
    for (int32_t ax0 = 0; ax0 < 2; ax0 += 1) {
      for (int32_t ax1 = 0; ax1 < 64; ax1 += 1) {
        B[((64 * ax0) + ((128 * i_j_fused_0) + ax1))] = A[((64 * ax0) + ((128 * i_j_fused_0) + ax1))];
      };
    };
    for (int32_t i_j_fused_1 = 0; i_j_fused_1 < 128; i_j_fused_1 += 1) {
      C[((128 * i_j_fused_0) + i_j_fused_1)] = B[((128 * i_j_fused_0) + i_j_fused_1)];
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

#ifdef CINN_WITH_CUDA
TEST(IrSchedule, compute_at4) {
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

  auto func = cinn::lang::LowerVec("test_compute_at4", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

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

  VLOG(1) << "compute_at4 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
typedef long int int64_t;
#endif



__global__
void test_compute_at4(const float* __restrict__ A, float* __restrict__ C)
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

TEST(IrSchedule, compute_at5) {
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

  auto func = cinn::lang::LowerVec("test_compute_at5", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

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

  VLOG(1) << "compute_at5 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
typedef long int int64_t;
#endif



__global__
void test_compute_at5(const float* __restrict__ A, float* __restrict__ C)
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

TEST(IrSchedule, compute_at6) {
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

  auto func = cinn::lang::LowerVec("test_compute_at6", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

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

  VLOG(1) << "compute_at6 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
typedef long int int64_t;
#endif



__global__
void test_compute_at6(const float* __restrict__ A, float* __restrict__ C)
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
#endif

TEST(IrSchedule, cache_read1) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);
  Expr P(16);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {N, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {P, P}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_cache_read1", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto a_cache = ir_sch.CacheRead(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto b_cache = ir_sch.CacheRead(block_c, 0, "local");

  VLOG(1) << "After CacheRead, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_read1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_read1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  {
    for (int32_t cache_ax0 = 0; cache_ax0 < 32; cache_ax0 += 1) {
      for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
        A_local_temp_buffer[((64 * cache_ax0) + cache_ax1)] = A[((64 * cache_ax0) + cache_ax1)];
      };
    };
    for (int32_t i = 0; i < 32; i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        B[((32 * i) + j)] = (2 * A_local_temp_buffer[((64 * i) + j)]);
      };
    };
    for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 16; cache_ax0_0 += 1) {
      for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 16; cache_ax1_0 += 1) {
        B_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)] = B[((32 * cache_ax0_0) + cache_ax1_0)];
      };
    };
    for (int32_t i = 0; i < 16; i += 1) {
      for (int32_t j = 0; j < 16; j += 1) {
        C[((16 * i) + j)] = (1 + B_local_temp_buffer[((32 * i) + j)]);
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, cache_read2) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec("test_cache_read2", stages, {A, B}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");

  auto a_cache = ir_sch.CacheRead(block_b, 0, "local");

  auto loops = ir_sch.GetLoops("B");
  ir_sch.ComputeAt(a_cache, loops[1]);

  VLOG(1) << "After CacheRead and ComputeAt, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_read2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_read2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      A_local_temp_buffer[((32 * i) + j)] = A[((32 * i) + j)];
      B[((32 * i) + j)] = (2 * A_local_temp_buffer[((32 * i) + j)]);
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, cache_write1) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_cache_write1", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto c_cache = ir_sch.CacheWrite(block_c, 0, "local");

  VLOG(1) << "After CacheWrite, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_write1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_write1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 64, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  {
    for (int32_t i = 0; i < 64; i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        B_local_temp_buffer[((32 * i) + j)] = (2 * A[((32 * i) + j)]);
      };
    };
    for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
      for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
        B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
      };
    };
    for (int32_t i = 0; i < 64; i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        C_local_temp_buffer[((32 * i) + j)] = (1 + B[((32 * i) + j)]);
      };
    };
    for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 64; cache_ax0_0 += 1) {
      for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 32; cache_ax1_0 += 1) {
        C[((32 * cache_ax0_0) + cache_ax1_0)] = C_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)];
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, cache_write2) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec("test_cache_write2", stages, {A, B}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto loops   = ir_sch.GetLoops("B");
  ir_sch.ComputeAt(b_cache, loops[1]);

  VLOG(1) << "After CacheWrite and ComputeAt, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_write2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_write2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
    for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
      B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)] = (2 * A[((32 * cache_ax0) + cache_ax1)]);
      B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

#ifdef CINN_WITH_CUDA
TEST(IrSchedule, cache_read3) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);
  Expr P(16);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {N, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {P, P}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec("test_cache_read3", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto a_cache = ir_sch.CacheRead(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto b_cache = ir_sch.CacheRead(block_c, 0, "local");
  auto loops_c = ir_sch.GetLoops("C");
  ir_sch.SyncThreads(loops_c[1], false);
  auto loops_b = ir_sch.GetLoops("B");
  ir_sch.SyncThreads(loops_b[1]);

  VLOG(1) << "After CacheRead, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_read3 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
typedef long int int64_t;
#endif



__global__
void test_cache_read3(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 1024 ];
  float* B = _B_temp_buffer;
  {
    for (int32_t cache_ax0 = 0; cache_ax0 < 32; cache_ax0 += 1) {
      for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
        A_local_temp_buffer[((64 * cache_ax0) + cache_ax1)] = A[((64 * cache_ax0) + cache_ax1)];
      };
    };
    for (int32_t i = 0; i < 32; i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        B[((32 * i) + j)] = (2 * A_local_temp_buffer[((64 * i) + j)]);
      };
      __syncthreads();
    };
    for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 16; cache_ax0_0 += 1) {
      for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 16; cache_ax1_0 += 1) {
        B_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)] = B[((32 * cache_ax0_0) + cache_ax1_0)];
      };
    };
    for (int32_t i = 0; i < 16; i += 1) {
      __syncthreads();
      for (int32_t j = 0; j < 16; j += 1) {
        C[((16 * i) + j)] = (1 + B_local_temp_buffer[((32 * i) + j)]);
      };
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, cache_write3) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("shared");

  auto func = cinn::lang::LowerVec("test_cache_write3", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto c_cache = ir_sch.CacheWrite(block_c, 0, "local");
  auto loops_c = ir_sch.GetLoops("C");
  ir_sch.SyncThreads(loops_c[0], false);
  auto loops_b = ir_sch.GetLoops("B");
  ir_sch.SyncThreads(loops_b[0]);

  VLOG(1) << "After CacheWrite, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_write3 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
typedef long int int64_t;
#endif



__global__
void test_cache_write3(const float* __restrict__ A, float* __restrict__ C)
{
  __shared__ float _B_temp_buffer [ 2048 ];
  float* B = _B_temp_buffer;
  {
    for (int32_t i = 0; i < 64; i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        B_local_temp_buffer[((32 * i) + j)] = (2 * A[((32 * i) + j)]);
      };
    };
    for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
      for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
        B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
      };
    };
    __syncthreads();
    for (int32_t i = 0; i < 64; i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        C_local_temp_buffer[((32 * i) + j)] = (1 + B[((32 * i) + j)]);
      };
    };
    __syncthreads();
    for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 64; cache_ax0_0 += 1) {
      for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 32; cache_ax1_0 += 1) {
        C[((32 * cache_ax0_0) + cache_ax1_0)] = C_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)];
      };
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, sync_threads) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("shared");

  auto func = cinn::lang::LowerVec("test_sync_threads", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto c_cache = ir_sch.CacheWrite(block_c, 0, "local");
  block_c      = ir_sch.GetBlock("C");
  ir_sch.SyncThreads(block_c, false);
  block_b = ir_sch.GetBlock("B");
  ir_sch.SyncThreads(block_b);

  VLOG(1) << "After CacheWrite and SyncThreads, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "sync_threads source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
typedef long int int64_t;
#endif



__global__
void test_sync_threads(const float* __restrict__ A, float* __restrict__ C)
{
  __shared__ float _B_temp_buffer [ 2048 ];
  float* B = _B_temp_buffer;
  {
    for (int32_t i = 0; i < 64; i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        B_local_temp_buffer[((32 * i) + j)] = (2 * A[((32 * i) + j)]);
      };
    };
    for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
      for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
        B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
        __syncthreads();
      };
    };
    for (int32_t i = 0; i < 64; i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        C_local_temp_buffer[((32 * i) + j)] = (1 + B[((32 * i) + j)]);
      };
    };
    for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 64; cache_ax0_0 += 1) {
      for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 32; cache_ax1_0 += 1) {
        __syncthreads();
        C[((32 * cache_ax0_0) + cache_ax1_0)] = C_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)];
      };
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}
#endif

TEST(IrSchedule, rfactor) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);
  Expr K(16);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, K});
  Var j(2, "j0");
  Var k(16, "k0");
  auto B = Compute(
      {M},
      [&](Var i) {
        return lang::ReduceSum(A(i, j, k), {j, k});
      },
      "B");

  auto stages = CreateStages({A, B});
  auto func   = cinn::lang::LowerVec("test_rfactor", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 3U);
  auto new_rf_tensor      = ir_sch.Rfactor(loops[2], 0);
  auto* new_rf_tensor_ref = new_rf_tensor.As<ir::_Tensor_>();
  CHECK(new_rf_tensor_ref);
  CHECK(new_rf_tensor_ref->buffer.defined());
  func[0]->temp_bufs.push_back(new_rf_tensor_ref->buffer);
  func[0]->PrepareBufferCastExprs();
  std::string origin = utils::GetStreamCnt(func[0]);
  LOG(INFO) << "After rfactor , func is : \n" << func[0];
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_rfactor (_A, _B)
{
  ScheduleBlock(root)
  {
    {
      serial for (rf_k0, 0, 16)
      {
        serial for (i, 0, 32)
        {
          ScheduleBlock(rf_B__reduce_init)
          {
            i0, i1 = axis.bind(i, rf_k0)
            rf_B__reduce_init[i1, i0] = 0
          }
          serial for (j0, 0, 2)
          {
            ScheduleBlock(rf_B)
            {
              i0, i1, i2 = axis.bind(i, j0, rf_k0)
              rf_B[i2, i0] = (rf_B[i2, i0] + A[i0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        ScheduleBlock(B__reduce_init)
        {
          i0 = axis.bind(i)
          B__reduce_init[i0] = 0
        }
        serial for (k0, 0, 16)
        {
          ScheduleBlock(B)
          {
            i0, i2 = axis.bind(i, k0)
            B[i0] = (B[i0] + rf_B[i2, i0])
          }
        }
      }
    }
  }
}
)ROC"));
  // optimze pass: add temp buffers
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "rfactor source code is :\n" << source_code;
  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_rfactor(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* rf__B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 16, 32 });
  cinn_buffer_malloc((void*)(0), _B);
  cinn_buffer_malloc((void*)(0), rf__B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* B__reduce_init = ((float*)(_B->memory));
  float* rf_B = ((float*)(rf__B->memory));
  float* rf_B__reduce_init = ((float*)(rf__B->memory));
  {
    for (int32_t rf_k0 = 0; rf_k0 < 16; rf_k0 += 1) {
      for (int32_t i = 0; i < 32; i += 1) {
        rf_B__reduce_init[((32 * rf_k0) + i)] = 0;
        for (int32_t j0 = 0; j0 < 2; j0 += 1) {
          rf_B[((32 * rf_k0) + i)] = (rf_B[((32 * rf_k0) + i)] + A[((32 * i) + ((16 * j0) + rf_k0))]);
        };
      };
    };
    for (int32_t i = 0; i < 32; i += 1) {
      B__reduce_init[i] = 0;
      for (int32_t k0 = 0; k0 < 16; k0 += 1) {
        B[i] = (B[i] + rf_B[((32 * k0) + i)]);
      };
    };
  };
  cinn_buffer_free((void*)(0), rf__B);
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, rfactor1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);
  Expr K(16);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, K});
  Var j(2, "j0");
  Var k(16, "k0");
  auto B = Compute(
      {M},
      [&](Var i) {
        return lang::ReduceSum(A(i, j, k), {j, k});
      },
      "B");

  auto stages = CreateStages({A, B});
  auto func   = cinn::lang::LowerVec("test_rfactor", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 3U);
  auto new_rf_tensor      = ir_sch.Rfactor(loops[1], 1);
  auto* new_rf_tensor_ref = new_rf_tensor.As<ir::_Tensor_>();
  CHECK(new_rf_tensor_ref);
  CHECK(new_rf_tensor_ref->buffer.defined());
  func[0]->temp_bufs.push_back(new_rf_tensor_ref->buffer);
  func[0]->PrepareBufferCastExprs();
  std::string origin = utils::GetStreamCnt(func[0]);
  LOG(INFO) << "After rfactor , func is : \n" << func[0];
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_rfactor (_A, _B)
{
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (rf_j0, 0, 2)
        {
          ScheduleBlock(rf_B__reduce_init)
          {
            i0, i1 = axis.bind(i, rf_j0)
            rf_B__reduce_init[i0, i1] = 0
          }
          serial for (k0, 0, 16)
          {
            ScheduleBlock(rf_B)
            {
              i0, i1, i2 = axis.bind(i, rf_j0, k0)
              rf_B[i0, i1] = (rf_B[i0, i1] + A[i0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        ScheduleBlock(B__reduce_init)
        {
          i0 = axis.bind(i)
          B__reduce_init[i0] = 0
        }
        serial for (j0, 0, 2)
        {
          ScheduleBlock(B)
          {
            i0, i1 = axis.bind(i, j0)
            B[i0] = (B[i0] + rf_B[i0, i1])
          }
        }
      }
    }
  }
}
)ROC"));
  // optimze pass: add temp buffers
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "rfactor source code is :\n" << source_code;
  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_rfactor(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* rf__B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 2 });
  cinn_buffer_malloc((void*)(0), _B);
  cinn_buffer_malloc((void*)(0), rf__B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* B__reduce_init = ((float*)(_B->memory));
  float* rf_B = ((float*)(rf__B->memory));
  float* rf_B__reduce_init = ((float*)(rf__B->memory));
  {
    for (int32_t i = 0; i < 32; i += 1) {
      for (int32_t rf_j0 = 0; rf_j0 < 2; rf_j0 += 1) {
        rf_B__reduce_init[((2 * i) + rf_j0)] = 0;
        for (int32_t k0 = 0; k0 < 16; k0 += 1) {
          rf_B[((2 * i) + rf_j0)] = (rf_B[((2 * i) + rf_j0)] + A[((32 * i) + ((16 * rf_j0) + k0))]);
        };
      };
    };
    for (int32_t i = 0; i < 32; i += 1) {
      B__reduce_init[i] = 0;
      for (int32_t j0 = 0; j0 < 2; j0 += 1) {
        B[i] = (B[i] + rf_B[((2 * i) + j0)]);
      };
    };
  };
  cinn_buffer_free((void*)(0), rf__B);
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, rfactor2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);
  Expr K(16);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});
  Var k(16, "k0");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return lang::ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  auto stages = CreateStages({A, B, C});
  auto func   = cinn::lang::LowerVec("test_rfactor", stages, {A, B, C}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("C");
  CHECK_EQ(loops.size(), 3U);
  auto new_rf_tensor      = ir_sch.Rfactor(loops[2], 0);
  auto* new_rf_tensor_ref = new_rf_tensor.As<ir::_Tensor_>();
  CHECK(new_rf_tensor_ref);
  CHECK(new_rf_tensor_ref->buffer.defined());
  func[0]->temp_bufs.push_back(new_rf_tensor_ref->buffer);
  func[0]->PrepareBufferCastExprs();
  std::string origin = utils::GetStreamCnt(func[0]);
  LOG(INFO) << "After rfactor , func is : \n" << func[0];
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_rfactor (_A, _B, _C)
{
  ScheduleBlock(root)
  {
    {
      serial for (rf_k0, 0, 16)
      {
        serial for (i, 0, 32)
        {
          serial for (j, 0, 2)
          {
            ScheduleBlock(rf_C__reduce_init)
            {
              i0, i1, i2 = axis.bind(i, j, rf_k0)
              rf_C__reduce_init[i2, i0, i1] = 0
            }
            ScheduleBlock(rf_C)
            {
              i0, i1, i2 = axis.bind(i, j, rf_k0)
              rf_C[i2, i0, i1] = (rf_C[i2, i0, i1] + (A[i0, i2] * B[i2, i1]))
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 2)
        {
          ScheduleBlock(C__reduce_init)
          {
            i0, i1 = axis.bind(i, j)
            C__reduce_init[i0, i1] = 0
          }
          serial for (k0, 0, 16)
          {
            ScheduleBlock(C)
            {
              i0, i1, i2 = axis.bind(i, j, k0)
              C[i0, i1] = (C[i0, i1] + rf_C[i2, i0, i1])
            }
          }
        }
      }
    }
  }
}
)ROC"));
  // optimze pass: add temp buffers
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "rfactor source code is :\n" << source_code;
  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_rfactor(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_t* rf__C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 16, 32, 2 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), rf__C);
  const float* A = ((const float*)(_A->memory));
  const float* B = ((const float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  float* C__reduce_init = ((float*)(_C->memory));
  float* rf_C = ((float*)(rf__C->memory));
  float* rf_C__reduce_init = ((float*)(rf__C->memory));
  {
    for (int32_t rf_k0 = 0; rf_k0 < 16; rf_k0 += 1) {
      for (int32_t i = 0; i < 32; i += 1) {
        for (int32_t j = 0; j < 2; j += 1) {
          rf_C__reduce_init[((2 * i) + ((64 * rf_k0) + j))] = 0;
          rf_C[((2 * i) + ((64 * rf_k0) + j))] = fma(A[((16 * i) + rf_k0)], B[((2 * rf_k0) + j)], rf_C[((2 * i) + ((64 * rf_k0) + j))]);
        };
      };
    };
    for (int32_t i = 0; i < 32; i += 1) {
      for (int32_t j = 0; j < 2; j += 1) {
        C__reduce_init[((2 * i) + j)] = 0;
        for (int32_t k0 = 0; k0 < 16; k0 += 1) {
          C[((2 * i) + j)] = (C[((2 * i) + j)] + rf_C[((2 * i) + ((64 * k0) + j))]);
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), rf__C);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_inline1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(j, i, k) * Expr(2.f); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_compute_inline1", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeInline(block_b);
  VLOG(1) << "After ComputeInline, IR is : " << ir_sch.GetModule().GetExprs().at(0);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_inline1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_inline1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = (2 * (1 + A[((32 * i) + ((1024 * j) + k))]));
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_inline2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(i, j, k) * Expr(2.f); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_compute_inline2", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops   = ir_sch.GetLoops("C");
  ir_sch.ComputeAt(block_b, loops[1]);
  block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeInline(block_b);
  VLOG(1) << "After ComputeInline, IR is : " << ir_sch.GetModule().GetExprs().at(0);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_inline2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_inline2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = (2 * (1 + A[((1024 * i) + ((32 * j) + k))]));
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

#ifdef CINN_WITH_CUDA
TEST(IrSchedule, compute_inline3) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(j, i, k) * Expr(2.f); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec("test_compute_inline3", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeInline(block_b);
  VLOG(1) << "After ComputeInline, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_inline3 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
typedef long int int64_t;
#endif



__global__
void test_compute_inline3(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 32768 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = (2 * (1 + A[((32 * i) + ((1024 * j) + k))]));
      };
    };
  };
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_inline4) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(i, j, k) * Expr(2.f); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec("test_compute_inline4", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops   = ir_sch.GetLoops("C");
  ir_sch.ComputeAt(block_b, loops[1]);
  block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeInline(block_b);
  VLOG(1) << "After ComputeInline, IR is : " << ir_sch.GetModule().GetExprs().at(0);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "compute_inline4 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
typedef long int int64_t;
#endif



__global__
void test_compute_inline4(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 32768 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = (2 * (1 + A[((1024 * i) + ((32 * j) + k))]));
      };
    };
  };
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}
#endif

TEST(IrSchedule, copytransform1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(j, i, k) * Expr(2.f); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_copytransform1", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_c = ir_sch.GetBlock("C");
  auto loops_c = ir_sch.GetLoops(block_c);
  auto splited = ir_sch.Split(loops_c[1], {-1, 4});
  block_c      = ir_sch.GetBlock("C");
  loops_c      = ir_sch.GetLoops(block_c);
  splited      = ir_sch.Split(loops_c[0], {-1, 8});

  auto block_b = ir_sch.GetBlock("B");
  block_c      = ir_sch.GetBlock("C");

  LOG(INFO) << "Before CopyTransformAndLoopInfo, IR is : " << ir_sch.GetModule().GetExprs().at(0);
  ir_sch.CopyTransformAndLoopInfo(block_b, block_c);
  LOG(INFO) << "After CopyTransformAndLoopInfo, IR is : " << ir_sch.GetModule().GetExprs().at(0);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "copytransform1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_copytransform1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  {
    for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
      for (int32_t i_1 = 0; i_1 < 8; i_1 += 1) {
        for (int32_t j_0 = 0; j_0 < 8; j_0 += 1) {
          for (int32_t j_1 = 0; j_1 < 4; j_1 += 1) {
            for (int32_t k = 0; k < 32; k += 1) {
              B[((8192 * i_0) + ((1024 * i_1) + ((128 * j_0) + ((32 * j_1) + k))))] = (1 + A[((8192 * i_0) + ((1024 * i_1) + ((128 * j_0) + ((32 * j_1) + k))))]);
            };
          };
        };
      };
    };
    for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
      for (int32_t i_1 = 0; i_1 < 8; i_1 += 1) {
        for (int32_t j_0 = 0; j_0 < 8; j_0 += 1) {
          for (int32_t j_1 = 0; j_1 < 4; j_1 += 1) {
            for (int32_t k = 0; k < 32; k += 1) {
              C[((8192 * i_0) + ((1024 * i_1) + ((128 * j_0) + ((32 * j_1) + k))))] = (2 * B[((256 * i_0) + ((32 * i_1) + ((4096 * j_0) + ((1024 * j_1) + k))))]);
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, copytransform2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(64);
  Expr P(128);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); }, "B");
  auto C = Compute(
      {M, M, P}, [&](Var i, Var j, Var k) { return B(i, j, k) * Expr(2.f); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec("test_copytransform2", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_c = ir_sch.GetBlock("C");
  auto loops_c = ir_sch.GetLoops(block_c);
  auto splited = ir_sch.Split(loops_c[1], {-1, 4});
  block_c      = ir_sch.GetBlock("C");
  loops_c      = ir_sch.GetLoops(block_c);
  splited      = ir_sch.Split(loops_c[0], {-1, 8});

  auto block_b = ir_sch.GetBlock("B");
  block_c      = ir_sch.GetBlock("C");
  ir_sch.CopyTransformAndLoopInfo(block_b, block_c);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  LOG(INFO) << "copytransform2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_copytransform2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 64, 128 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  {
    for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
      for (int32_t i_1 = 0; i_1 < 8; i_1 += 1) {
        for (int32_t j = 0; j < 64; j += 1) {
          for (int32_t k = 0; k < 128; k += 1) {
            B[((65536 * i_0) + ((8192 * i_1) + ((128 * j) + k)))] = (1 + A[((65536 * i_0) + ((8192 * i_1) + ((128 * j) + k)))]);
          };
        };
      };
    };
    for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
      for (int32_t i_1 = 0; i_1 < 8; i_1 += 1) {
        for (int32_t j_0 = 0; j_0 < 8; j_0 += 1) {
          for (int32_t j_1 = 0; j_1 < 4; j_1 += 1) {
            for (int32_t k = 0; k < 128; k += 1) {
              C[((32768 * i_0) + ((4096 * i_1) + ((512 * j_0) + ((128 * j_1) + k))))] = (2 * B[((65536 * i_0) + ((8192 * i_1) + ((512 * j_0) + ((128 * j_1) + k))))]);
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, Annotate) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto funcs = cinn::lang::LowerVec(
      "test_split_and_fuse1", CreateStages({A, B}), {A, B}, {}, {}, nullptr, common::DefaultHostTarget(), true);
  ir::IRSchedule ir_sch(ir::ModuleExpr({funcs[0]->body}));
  auto fused   = ir_sch.Fuse("B", {0, 1});
  auto block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k1", int(64));
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k2", bool(true));
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k3", float(2.0));
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k4", std::string("v4"));
  std::string expected_expr = R"ROC({
  ScheduleBlock(root)
  {
    serial for (i_j_fused, 0, 1024)
    {
      ScheduleBlock(B)
      {
        i0, i1 = axis.bind((i_j_fused / 32), (i_j_fused % 32))
        attrs(k1:64, k2:1, k3:2, k4:v4)
        B[i0, i1] = A[i0, i1]
      }
    }
  }
})ROC";
  ASSERT_EQ(utils::GetStreamCnt(ir_sch.GetModule().GetExprs().front()), expected_expr);
}

}  // namespace backends
}  // namespace cinn

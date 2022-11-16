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

#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_read.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/compiler.h"
#include "cinn/cinn.h"
#include "cinn/common/cuda_test_helper.h"
#include "cinn/common/test_helper.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/module.h"
#include "cinn/ir/tensor.h"
#include "cinn/poly/stage.h"
#ifdef CINN_WITH_CUDA
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"
#endif
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

void naive_matmul(float* A, float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void check_matmul_result_cpu(int M, int N, int K, void (*func_ptr)(void**, int32_t)) {
  // prepare data
  auto* A_host = common::BufferBuilder(Float(32), {M, N}).set_random().Build();
  CHECK(A_host);
  auto* B_host = common::BufferBuilder(Float(32), {M, N}).set_random().Build();
  CHECK(B_host);
  auto* C_host = common::BufferBuilder(Float(32), {M, N}).set_zero().Build();
  CHECK(C_host);
  auto all_args = common::ArgsBuilder().Add(A_host).Add(B_host).Add(C_host).Build();

  // calculate matmul after schedule in rule
  func_ptr(reinterpret_cast<void**>(all_args.data()), all_args.size());
  float* res = reinterpret_cast<float*>(C_host->memory);

  // calculate naive matmul
  float* data_A     = reinterpret_cast<float*>(A_host->memory);
  float* data_B     = reinterpret_cast<float*>(B_host->memory);
  float* target_res = (float*)malloc(M * N * sizeof(float));
  memset(target_res, 0, M * N * sizeof(float));
  naive_matmul(data_A, data_B, target_res, M, N, K);

  // check result
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      ASSERT_NEAR(res[i * N + j], target_res[i * N + j], 1e-4);
    }
  }

  cinn_buffer_free(nullptr, A_host);
  cinn_buffer_free(nullptr, B_host);
  cinn_buffer_free(nullptr, C_host);
  free(target_res);
}

#ifdef CINN_WITH_CUDA
void* CreateDeviceBuffer(const cinn_buffer_t* host_buffer) {
  CHECK(host_buffer->memory);
  int num_bytes = host_buffer->num_elements() * sizeof(float);
  VLOG(6) << "create device buffer, num_bytes = " << num_bytes;
  CUdeviceptr data;
  cuMemAlloc(&data, num_bytes);

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(data), host_buffer->memory, num_bytes, cudaMemcpyHostToDevice));
  return reinterpret_cast<void*>(data);
}

void check_matmul_result_cuda(int M, int N, int K, void (*func_ptr)(void**, int32_t)) {
  // prepare data
  auto* A_host = common::BufferBuilder(Float(32), {M, N}).set_random().Build();
  CHECK(A_host);
  auto* B_host = common::BufferBuilder(Float(32), {M, N}).set_random().Build();
  CHECK(B_host);
  auto* C_host = common::BufferBuilder(Float(32), {M, N}).set_zero().Build();
  CHECK(C_host);

  auto* A_dev = CreateDeviceBuffer(A_host);
  auto* B_dev = CreateDeviceBuffer(B_host);
  auto* C_dev = CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[3];
  for (int i = 0; i < 3; ++i) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->memory = reinterpret_cast<uint8_t*>(C_dev);
  auto all_args       = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  // calculate matmul after schedule in rule
  CUDA_CALL(cudaDeviceSynchronize());
  CHECK(func_ptr);
  func_ptr(reinterpret_cast<void**>(all_args.data()), all_args.size());
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(
      reinterpret_cast<void*>(C_host->memory), C_dev, C_host->num_elements() * sizeof(float), cudaMemcpyDeviceToHost));
  float* res = reinterpret_cast<float*>(C_host->memory);

  // calculate common matmul
  float* data_A     = reinterpret_cast<float*>(A_host->memory);
  float* data_B     = reinterpret_cast<float*>(B_host->memory);
  float* target_res = (float*)malloc(M * N * sizeof(float));
  memset(target_res, 0, M * N * sizeof(float));
  naive_matmul(data_A, data_B, target_res, M, N, K);

  // check result
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      ASSERT_NEAR(res[i * N + j], target_res[i * N + j], 1e-4);
    }
  }

  cinn_buffer_free(nullptr, A_host);
  cinn_buffer_free(nullptr, B_host);
  cinn_buffer_free(nullptr, C_host);
  free(target_res);
  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(B_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}
#endif

TEST(AddCacheRead, Init) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ir::Expr M(32);
  ir::Expr N(32);
  ir::Expr K(32);

  // matmul case
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestAddCacheRead_InitTrue", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr matmul_expr = funcs[0]->body;
  VLOG(6) << "Matmul Expr before AddCacheRead: ";
  VLOG(6) << matmul_expr;

  ir::IRSchedule ir_schedule_matmul(ir::ModuleExpr({matmul_expr}));

  AddCacheRead add_cache_read(target);
  EXPECT_EQ(add_cache_read.Init(&ir_schedule_matmul), RuleApplyType::kApplyAndSkipAllRules);
  EXPECT_EQ(add_cache_read.NumberApplicable(), 1);

  add_cache_read.ApplyRandomly();
  std::vector<ir::Expr> exprs = ir_schedule_matmul.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Matmul Expr after AddCacheRead: " << exprs[0];

  // add case
  Placeholder<float> D("D", {M, K});
  Placeholder<float> E("E", {K, N});
  ir::Tensor F = Compute(
      {M, N}, [&](Var i, Var j) { return D(i, j) + E(i, j); }, "F");

  poly::StageMap stages_add = CreateStages({F});
  std::vector<ir::LoweredFunc> funcs_add =
      lang::LowerVec("TestAddCacheRead_InitFalse", stages_add, {F}, {}, {}, nullptr, target, true);

  ir::Expr add_expr = funcs_add[0]->body;
  VLOG(6) << "Mat Add Expr before AddCacheRead: ";
  VLOG(6) << add_expr;

  ir::IRSchedule ir_schedule_add(ir::ModuleExpr({add_expr}));

  AddCacheRead add_cache_read2(target);
  EXPECT_EQ(add_cache_read2.Init(&ir_schedule_add), RuleApplyType::kCannotApply);
  EXPECT_EQ(add_cache_read2.NumberApplicable(), 0);
}

TEST(AddCacheRead, MatrixMultiply) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ir::Expr M(32);
  ir::Expr N(32);
  ir::Expr K(32);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestAddCacheRead_MatrixMultiply", stages, {A, B, C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));

  // Apply MultiLevelTiling before AddCacheRead
  MultiLevelTiling multi_level_tiling(target);
  EXPECT_EQ(multi_level_tiling.Init(&ir_schedule), RuleApplyType::kApplyAndSkipThisRule);
  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);

  multi_level_tiling.ApplyRandomly();
  std::vector<ir::Expr> exprs = ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after MultiLevelTiling: " << exprs[0];

  // Apply AddCacheRead
  AddCacheRead add_cache_read(target);
  EXPECT_EQ(add_cache_read.Init(&ir_schedule), RuleApplyType::kApplyAndSkipAllRules);
  EXPECT_EQ(add_cache_read.NumberApplicable(), 1);

  add_cache_read.ApplyRandomly();
  exprs = ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after AddCacheRead: " << exprs[0];

  auto temp_buffers = lang::GetTempBuffers({A, B, C}, stages, exprs[0]);
  auto func         = ir::_LoweredFunc_::Make(funcs[0]->name, funcs[0]->args, exprs[0], temp_buffers);

  ir::Module::Builder builder("test_bulder", target);
  builder.AddFunction(func);
  auto build_module = builder.Build();

  auto compiler = backends::Compiler::Create(target);
  compiler->Build(build_module);

  // print source code for debug
  backends::CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(build_module, CodeGenC::OutputKind::CImpl);
  VLOG(6) << "source code is :\n" << source_code;

  // TODO(BiynXu): Debug and add accuracy test
  //   auto test_func_ptr = reinterpret_cast<void (*)(void**,
  //   int32_t)>(compiler->Lookup("TestAddCacheRead_MatrixMultiply"));

  // #ifdef CINN_WITH_CUDA
  //   check_matmul_result_cuda(M.as_int32(), N.as_int32(), K.as_int32(), test_func_ptr);
  // #else
  //   check_matmul_result_cpu(M.as_int32(), N.as_int32(), K.as_int32(), test_func_ptr);
  // #endif
}

}  // namespace auto_schedule
}  // namespace cinn
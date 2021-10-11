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

#include "cinn/backends/codegen_cuda_dev.h"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <tuple>
#include <vector>

#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/common/cuda_test_helper.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/lower.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"
#include "cinn/runtime/use_extern_funcs.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace backends {

std::tuple<CUdeviceptr, CUdeviceptr, CUdeviceptr, std::vector<float>, std::vector<float>, std::vector<float>>
CreateNVMemory(int M, int N) {
  CUDA_CALL(cudaDeviceSynchronize());

  CUdeviceptr Ad, Bd, Cd;
  cuMemAlloc(&Ad, M * N * sizeof(float));
  cuMemAlloc(&Bd, M * N * sizeof(float));
  cuMemAlloc(&Cd, M * N * sizeof(float));

  int num_elements = M * N;

  std::vector<float> host_data1(num_elements, 0);
  std::vector<float> host_data2(num_elements, 0);
  std::vector<float> host_data3(num_elements, 0);
  for (float& v : host_data1) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT
  for (float& v : host_data2) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT

  CUDA_CALL(
      cudaMemcpy(reinterpret_cast<void*>(Ad), host_data1.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(reinterpret_cast<void*>(Bd), host_data2.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

  return std::make_tuple(Ad, Bd, Cd, host_data1, host_data2, host_data3);
}

TEST(CodeGenCUDA, basic) {
  Expr M(1);
  Expr N(200);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", stages, {A, B, C});

  auto compiled = codegen.Compile(func);

  std::cout << "test cout: " << compiled << std::endl;
}

TEST(CodeGenCUDA, Module_output) {
  Expr M(100);
  Expr N(200);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", stages, {A, B, C});

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  Outputs outputs;
  outputs = outputs.cuda_source("_generated1.cu");
  codegen.Compile(builder.Build(), outputs);
}

TEST(CodeGenCUDA2, test_of_cacheread) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(200);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("X", {M, N});
  Placeholder<float> B("Y", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});
  std::vector<ir::Tensor> readers{C};
  auto B_cache = stages[B]->CacheRead("local", readers, stages);
  auto A_cache = stages[A]->CacheRead("local", readers, stages);
  stages[C]->Split(0, 10);
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[A_cache]->ComputeAt(stages[C], 1);
  stages[B_cache]->ComputeAt(stages[C], 1);
  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", stages, {A, B, C});

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled test_of_cacheread code:\n\n\n" << source_code;

  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  auto _Ad_Bd_Cd_host_data1_host_data2_host_data3_ = CreateNVMemory(M.as_int32(), N.as_int32());
  auto& Ad                                         = std::get<0>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Bd                                         = std::get<1>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Cd                                         = std::get<2>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data1                                 = std::get<3>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data2                                 = std::get<4>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data3                                 = std::get<5>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(10, 1, 1);
  dim3 block(10, 1, 1);
  cuda_module.LaunchKernel(0, "elementwise_add", grid, block, args);

  CUDA_CALL(cudaMemcpy(host_data3.data(),
                       reinterpret_cast<void*>(Cd),
                       M.as_int32() * N.as_int32() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < M.as_int32(); i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      int offset = i * N.as_int32() + j;
      EXPECT_NEAR(host_data3[offset], host_data1[offset] * host_data2[offset], 1e-5);
    }
  }
}

TEST(CodeGenCUDA2, test_of_splitcudakernel) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(200);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("X", {M, N});
  Placeholder<float> B("Y", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto D = Compute(
      {M, N}, [&](Var i, Var j) { return C(i, j) + B(i, j); }, "D");

  auto stages = CreateStages({C, D});

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[D]->Bind(0, "blockIdx.x");
  stages[D]->Bind(1, "threadIdx.x");

  CodeGenCUDA_Dev codegen(target);

  auto func = lang::LowerVec("elementwise_add", stages, {A, B, C, D}, {}, {}, nullptr, target);

  Module::Builder builder("module", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }

  auto module = builder.Build();

  auto _host_module_device_module_ = SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module                = std::get<0>(_host_module_device_module_);
  auto& device_module              = std::get<1>(_host_module_device_module_);

  auto source_code = codegen.Compile(module);

  LOG(INFO) << "compiled test_of_splitcudakernel code:\n\n\n" << source_code;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void elementwise_add(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ C)
{
  if ((blockIdx.x < 100)) {
    if ((threadIdx.x < 200)) {
      C[((200 * blockIdx.x) + threadIdx.x)] = (X[((200 * blockIdx.x) + threadIdx.x)] * Y[((200 * blockIdx.x) + threadIdx.x)]);
    };
  };
}__global__
void elementwise_add_1(const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ C, float* __restrict__ D)
{
  if ((blockIdx.x < 100)) {
    if ((threadIdx.x < 200)) {
      D[((200 * blockIdx.x) + threadIdx.x)] = (C[((200 * blockIdx.x) + threadIdx.x)] + Y[((200 * blockIdx.x) + threadIdx.x)]);
    };
  };
}

}
)ROC";
  ASSERT_EQ(utils::Trim(source_target), source_code);
}

TEST(CodeGenCUDA2, test_of_splitouter) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(100);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("X", {M, N});
  Placeholder<float> B("Y", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});
  std::vector<ir::Tensor> readers{C};
  stages[C]->SplitOuter(0, 20);
  stages[C]->SplitOuter(2, 17);
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add_splitouter", stages, {A, B, C});

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled test_of_splitouter code:\n\n\n" << source_code;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void elementwise_add_splitouter(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ C)
{
  if ((blockIdx.x < 20)) {
    if ((threadIdx.x < 5)) {
      for (int32_t j_outer = 0; j_outer < 17; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < cinn_nvgpu_min_fp32(6, (100 + (-6 * j_outer))); j_inner += 1) {
          C[((500 * blockIdx.x) + ((6 * j_outer) + ((100 * threadIdx.x) + j_inner)))] = (X[((500 * blockIdx.x) + ((6 * j_outer) + ((100 * threadIdx.x) + j_inner)))] * Y[((500 * blockIdx.x) + ((6 * j_outer) + ((100 * threadIdx.x) + j_inner)))]);
        };
      };
    };
  };
}

}
)ROC";
  ASSERT_EQ(utils::Trim(source_target), source_code);

  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  auto _Ad_Bd_Cd_host_data1_host_data2_host_data3_ = CreateNVMemory(M.as_int32(), N.as_int32());
  auto& Ad                                         = std::get<0>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Bd                                         = std::get<1>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Cd                                         = std::get<2>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data1                                 = std::get<3>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data2                                 = std::get<4>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data3                                 = std::get<5>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(20, 1, 1);
  dim3 block(5, 1, 1);
  cuda_module.LaunchKernel(0, "elementwise_add_splitouter", grid, block, args);

  CUDA_CALL(cudaMemcpy(host_data3.data(),
                       reinterpret_cast<void*>(Cd),
                       M.as_int32() * N.as_int32() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < M.as_int32(); i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      int offset = i * N.as_int32() + j;
      EXPECT_NEAR(host_data3[offset], host_data1[offset] * host_data2[offset], 1e-5);
    }
  }
}

TEST(CodeGenCUDA2, test_schedule_conv2d_0) {
  Context::Global().ResetNameId();
  Expr N(1);
  Expr C(128);
  Expr H(28);
  Expr W(256);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("X", {N, C, H, H});
  Placeholder<float> B("Y", {W, C, N, N});

  auto res = hlir::pe::Conv2d_NCHW(A, B, 0, 0, 2, 2, 1, 1, "COD");

  auto stages = CreateStages(res);

  auto pad_data = res[1];
  auto conv     = res[0];
  auto B_t      = B.tensor();

  hlir::pe::CudaScheduleConv(stages, pad_data, B_t, conv, target);

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("schedule_conv2d_0", stages, {A, B, conv}, {}, {}, nullptr, target);

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled schedule_conv2d_0 code:\n\n\n" << source_code;

  std::string source_target        = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void schedule_conv2d_0(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ COD)
{
  __shared__ float _input_pad_0_read_cache [ 224 ];
  float _COD_write_cache [ 2 ];
  __shared__ float _Y_read_cache [ 256 ];
  float* COD_write_cache = _COD_write_cache;
  float* COD_write_cache__reduce_init = _COD_write_cache;
  float* Y_read_cache = _Y_read_cache;
  float* input_pad_0_read_cache = _input_pad_0_read_cache;
  if ((blockIdx.z < 8)) {
    if ((blockIdx.y < 14)) {
      if ((threadIdx.z < 16)) {
        if ((threadIdx.x < 14)) {
        {
          for (int32_t rc_outer = 0; rc_outer < 2; rc_outer += 1) {
            COD_write_cache__reduce_init[rc_outer] = 0;
          };
          for (int32_t rc_outer = 0; rc_outer < 16; rc_outer += 1) {
            {
              __syncthreads();
              if ((threadIdx.z < 8)) {
                input_pad_0_read_cache[((2 * threadIdx.x) + (28 * threadIdx.z))] = X[((56 * blockIdx.y) + ((6272 * rc_outer) + ((2 * threadIdx.x) + (784 * threadIdx.z))))];
              };
            };
            for (int32_t rc_inner = 0; rc_inner < 2; rc_inner += 1) {
              if ((threadIdx.x < 8)) {
                Y_read_cache[((threadIdx.x / 2) + ((8 * (threadIdx.x & 1)) + ((4 * rc_inner) + (16 * threadIdx.z))))] = Y[((threadIdx.x / 2) + ((128 * (threadIdx.x & 1)) + ((4096 * blockIdx.z) + ((4 * rc_inner) + ((8 * rc_outer) + (256 * threadIdx.z))))))];
              };
            };
            __syncthreads();
            for (int32_t rc_inner = 0; rc_inner < 8; rc_inner += 1) {
              for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
                COD_write_cache[j_inner] = (COD_write_cache[j_inner] + (input_pad_0_read_cache[((28 * rc_inner) + (2 * threadIdx.x))] * Y_read_cache[((8 * j_inner) + ((16 * threadIdx.z) + rc_inner))]));
              };
            };
          };
          for (int32_t rc_outer = 0; rc_outer < 2; rc_outer += 1) {
            COD[((14 * blockIdx.y) + ((6272 * blockIdx.z) + ((196 * rc_outer) + ((392 * threadIdx.z) + threadIdx.x))))] = COD_write_cache[rc_outer];
          };
        }
        };
      };
    };
  };
}

}
)ROC";
  std::string trimed_source_target = utils::Trim(source_target);
  int start_target                 = trimed_source_target.find("blockIdx");
  int start_source                 = source_code.find("blockIdx");
  ASSERT_EQ(trimed_source_target.substr(start_target), source_code.substr(start_source));
  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  CUDA_CALL(cudaDeviceSynchronize());

  CUdeviceptr Ad, Bd, Cd;
  cuMemAlloc(&Ad, 128 * 28 * 28 * sizeof(float));
  cuMemAlloc(&Bd, 256 * 128 * sizeof(float));
  cuMemAlloc(&Cd, 256 * 14 * 14 * sizeof(float));

  std::vector<float> host_data1(128 * 28 * 28, 0);
  std::vector<float> host_data2(256 * 128, 0);
  std::vector<float> host_data3(256 * 14 * 14, 0);
  for (float& v : host_data1) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT
  for (float& v : host_data2) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT

  CUDA_CALL(cudaMemcpy(
      reinterpret_cast<void*>(Ad), host_data1.data(), host_data1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(
      reinterpret_cast<void*>(Bd), host_data2.data(), host_data2.size() * sizeof(float), cudaMemcpyHostToDevice));

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(1, 14, 8);
  dim3 block(14, 1, 16);
  int repeat = 100;

  utils::Timer time1;
  time1.Start();
  for (int i = 0; i < repeat; i++) {
    cuda_module.LaunchKernel(0, "schedule_conv2d_0", grid, block, args);
  }
  LOG(INFO) << "Conv2d op with schedule repeats " << repeat
            << " times, average time cost is : " << time1.Stop() / static_cast<float>(repeat) << "ms. ";
  CUDA_CALL(cudaMemcpy(
      host_data3.data(), reinterpret_cast<void*>(Cd), 256 * 14 * 14 * sizeof(float), cudaMemcpyDeviceToHost));
}

TEST(CodeGenCUDA2, test_schedule_conv2d_1) {
  Context::Global().ResetNameId();
  Expr N(1);
  Expr C(3);
  Expr H(224);
  Expr W(64);
  Expr K(7);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("X", {N, C, H, H});
  Placeholder<float> B("Y", {W, C, K, K});

  auto res = hlir::pe::Conv2d_NCHW(A, B, 3, 3, 2, 2, 1, 1, "Conv2d_out");

  auto stages = CreateStages(res);

  auto pad_data = res[1];
  auto conv     = res[0];
  auto B_t      = B.tensor();

  hlir::pe::CudaScheduleConv(stages, pad_data, B_t, conv, target);

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("schedule_conv2d_1", stages, {A, B, conv}, {}, {}, nullptr, target);

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled schedule_conv2d_1 code:\n\n\n" << source_code;

  std::string source_target        = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void schedule_conv2d_1(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ Conv2d_out)
{
  float _Conv2d_out_write_cache [ 2 ];
  __shared__ float _input_pad_0_read_cache [ 76 ];
  __shared__ float _Y_read_cache [ 112 ];
  float* Conv2d_out_write_cache = _Conv2d_out_write_cache;
  float* Conv2d_out_write_cache__reduce_init = _Conv2d_out_write_cache;
  float* Y_read_cache = _Y_read_cache;
  float* input_pad_0_read_cache = _input_pad_0_read_cache;
  if ((blockIdx.y < 112)) {
    for (int32_t j_outer_outer_inner = 0; j_outer_outer_inner < 4; j_outer_outer_inner += 1) {
      for (int32_t a_outer_outer_inner = 0; a_outer_outer_inner < 7; a_outer_outer_inner += 1) {
        if ((threadIdx.z < 8)) {
          if ((threadIdx.x < 16)) {
          {
            for (int32_t rc_outer = 0; rc_outer < 2; rc_outer += 1) {
              Conv2d_out_write_cache__reduce_init[rc_outer] = 0;
            };
            for (int32_t rc_outer = 0; rc_outer < 3; rc_outer += 1) {
              for (int32_t ry_outer = 0; ry_outer < 7; ry_outer += 1) {
                {
                  __syncthreads();
                  if ((threadIdx.z < 7)) {
                    input_pad_0_read_cache[((2 * threadIdx.x) + threadIdx.z)] = ((((((((2 * blockIdx.y) + ry_outer) >= 3) && (((2 * blockIdx.y) + ry_outer) < 227)) && (((32 * a_outer_outer_inner) + ((2 * threadIdx.x) + threadIdx.z)) >= 3)) && (((32 * a_outer_outer_inner) + ((2 * threadIdx.x) + threadIdx.z)) < 227))) ? X[(-675 + ((32 * a_outer_outer_inner) + ((448 * blockIdx.y) + ((50176 * rc_outer) + ((224 * ry_outer) + ((2 * threadIdx.x) + threadIdx.z))))))] : 0);
                  };
                };
                if ((threadIdx.x < 14)) {
                  Y_read_cache[((threadIdx.x / 2) + ((7 * (threadIdx.x & 1)) + (14 * threadIdx.z)))] = Y[((threadIdx.x / 2) + ((147 * (threadIdx.x & 1)) + ((2352 * j_outer_outer_inner) + ((49 * rc_outer) + ((7 * ry_outer) + (294 * threadIdx.z))))))];
                };
                __syncthreads();
                for (int32_t rx_inner = 0; rx_inner < 7; rx_inner += 1) {
                  for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
                    Conv2d_out_write_cache[j_inner] = (Conv2d_out_write_cache[j_inner] + (input_pad_0_read_cache[((2 * threadIdx.x) + rx_inner)] * Y_read_cache[((7 * j_inner) + ((14 * threadIdx.z) + rx_inner))]));
                  };
                };
              };
            };
            for (int32_t rc_outer = 0; rc_outer < 2; rc_outer += 1) {
              Conv2d_out[((16 * a_outer_outer_inner) + ((112 * blockIdx.y) + ((200704 * j_outer_outer_inner) + ((12544 * rc_outer) + ((25088 * threadIdx.z) + threadIdx.x)))))] = Conv2d_out_write_cache[rc_outer];
            };
          }
          };
        };
      };
    };
  };
}

}
  )ROC";
  std::string trimed_source_target = utils::Trim(source_target);
  int start_target                 = trimed_source_target.find("blockIdx");
  int start_source                 = source_code.find("blockIdx");
  ASSERT_EQ(trimed_source_target.substr(start_target), source_code.substr(start_source));
  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  CUDA_CALL(cudaDeviceSynchronize());

  CUdeviceptr Ad, Bd, Cd;
  cuMemAlloc(&Ad, 1 * 3 * 224 * 224 * sizeof(float));
  cuMemAlloc(&Bd, 64 * 3 * 7 * 7 * sizeof(float));
  cuMemAlloc(&Cd, 1 * 64 * 112 * 112 * sizeof(float));

  std::vector<float> host_data1(1 * 3 * 224 * 224, 0);
  std::vector<float> host_data2(64 * 3 * 7 * 7, 0);
  std::vector<float> host_data3(1 * 64 * 112 * 112, 0);
  for (float& v : host_data1) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT
  for (float& v : host_data2) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT

  CUDA_CALL(cudaMemcpy(
      reinterpret_cast<void*>(Ad), host_data1.data(), host_data1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(
      reinterpret_cast<void*>(Bd), host_data2.data(), host_data2.size() * sizeof(float), cudaMemcpyHostToDevice));

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(1, 112, 1);
  dim3 block(16, 1, 8);
  int repeat = 5000;

  utils::Timer time1;
  time1.Start();
  for (int i = 0; i < repeat; i++) {
    cuda_module.LaunchKernel(0, "schedule_conv2d_1", grid, block, args);
    CUDA_CALL(cudaDeviceSynchronize());
  }
  auto time_average1 = time1.Stop() / static_cast<float>(repeat);
  LOG(INFO) << "Conv2d op1_CINN with schedule repeats " << repeat << " times, average time cost is : " << time_average1
            << "ms. ";
  CUDA_CALL(cudaMemcpy(
      host_data3.data(), reinterpret_cast<void*>(Cd), host_data3.size() * sizeof(float), cudaMemcpyDeviceToHost));

  std::string source_tvm = R"ROC(
  extern "C" {

  #include "cinn_cuda_runtime_source.cuh"

  #ifdef __CUDACC_RTC__
  typedef int int32_t;
  typedef char int8_t;
  #endif


  __global__ void schedule_conv2d_1(float* __restrict__ placeholder, float* __restrict__ placeholder1, float*
  __restrict__ Conv2d_out) { float compute[2];
    __shared__ float pad_temp_shared[37];
    __shared__ float placeholder_shared[112];
    for (int ax1_inner_outer = 0; ax1_inner_outer < 4; ++ax1_inner_outer) {
      for (int ax3_inner_outer = 0; ax3_inner_outer < 7; ++ax3_inner_outer) {
        for (int ff_init = 0; ff_init < 2; ++ff_init) {
          compute[(ff_init)] = 0.000000e+00f;
        }
        for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
          for (int ry_outer = 0; ry_outer < 7; ++ry_outer) {
            __syncthreads();
            if (((((int)threadIdx.z) * 5) + ((int)threadIdx.x)) < 37) {
              if (((int)threadIdx.x) < 5) {
                pad_temp_shared[(((((int)threadIdx.z) * 5) + ((int)threadIdx.x)))] = (((((3 <= ((((int)blockIdx.y) * 2)
  + ry_outer)) && (((((int)blockIdx.y) * 2) + ry_outer) < 227)) && (3 <= (((ax3_inner_outer * 32) + (((int)threadIdx.z)
  * 5)) + ((int)threadIdx.x)))) && ((((ax3_inner_outer * 32) + (((int)threadIdx.z) * 5)) + ((int)threadIdx.x)) < 227)) ?
  placeholder[((((((((rc_outer * 50176) + (((int)blockIdx.y) * 448)) + (ry_outer * 224)) + (ax3_inner_outer * 32)) +
  (((int)threadIdx.z) * 5)) + ((int)threadIdx.x)) - 675))] : 0.000000e+00f);
              }
            }
            if (((((int)threadIdx.z) * 2) + (((int)threadIdx.x) / 7)) < 16) {
              if (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) < 112) {
                if (((int)threadIdx.x) < 14) {
                  placeholder_shared[(((((int)threadIdx.z) * 14) + ((int)threadIdx.x)))] =
  placeholder1[(((((((ax1_inner_outer * 2352) + (((int)threadIdx.z) * 294)) + ((((int)threadIdx.x) / 7) * 147)) +
  (rc_outer * 49)) + (ry_outer * 7)) + (((int)threadIdx.x) % 7)))];
                }
              }
            }
            __syncthreads();
            for (int rx_inner = 0; rx_inner < 7; ++rx_inner) {
              for (int ff = 0; ff < 2; ++ff) {
                compute[(ff)] = (compute[(ff)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner))] *
  placeholder_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner))]));
              }
            }
          }
        }
        for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
          Conv2d_out[(((((((ax1_inner_outer * 200704) + (((int)threadIdx.z) * 25088)) + (ax1_inner_inner_inner * 12544))
  + (((int)blockIdx.y) * 112)) + (ax3_inner_outer * 16)) + ((int)threadIdx.x)))] = compute[(ax1_inner_inner_inner)];
        }
      }
    }
  }
  }
  )ROC";

  backends::NVRTC_Compiler compiler_tvm;

  auto ptx_tvm = compiler_tvm(source_tvm);
  CHECK(!ptx_tvm.empty());

  CUDAModule cuda_module_tvm(ptx_tvm, CUDAModule::Kind::PTX);

  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<float> host_data4(1 * 64 * 112 * 112, 0);
  // launch the kernel

  utils::Timer time2;
  time2.Start();
  for (int i = 0; i < repeat; i++) {
    cuda_module_tvm.LaunchKernel(0, "schedule_conv2d_1", grid, block, args);
    CUDA_CALL(cudaDeviceSynchronize());
  }
  auto time_average2 = time2.Stop() / static_cast<float>(repeat);
  LOG(INFO) << "Conv2d op1_TVM with schedule repeats " << repeat << " times, average time cost is : " << time_average2
            << "ms. ";
  CUDA_CALL(cudaMemcpy(
      host_data4.data(), reinterpret_cast<void*>(Cd), host_data4.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (int offset = 0; offset < host_data4.size(); offset++) {
    EXPECT_NEAR(host_data3[offset], host_data4[offset], 1e-5);
  }
}

TEST(CodeGenCUDA, test_of_syncthreads) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(200);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});
  std::vector<ir::Tensor> readers{C};
  auto B_cache = stages[B]->CacheRead("local", readers, stages);
  stages[B_cache]->Bind(0, "blockIdx.x");
  stages[B_cache]->Bind(1, "threadIdx.x");
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[B_cache]->SyncThreads(stages);
  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", stages, {A, B, C});

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled test_of_syncthreads code:\n\n\n" << source_code;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void elementwise_add(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  float _B_read_cache [ 1 ];
  float* B_read_cache = _B_read_cache;
  if ((blockIdx.x < 100)) {
    if ((threadIdx.x < 200)) {
      B_read_cache[0] = B[((200 * blockIdx.x) + threadIdx.x)];
    };
  };
  __syncthreads();
  if ((blockIdx.x < 100)) {
    if ((threadIdx.x < 200)) {
      C[((200 * blockIdx.x) + threadIdx.x)] = (A[((200 * blockIdx.x) + threadIdx.x)] * B_read_cache[0]);
    };
  };
}

}
)ROC";
  ASSERT_EQ(utils::Trim(source_target), source_code);

  // compile the code
  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  auto _Ad_Bd_Cd_host_data1_host_data2_host_data3_ = CreateNVMemory(M.as_int32(), N.as_int32());
  auto& Ad                                         = std::get<0>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Bd                                         = std::get<1>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Cd                                         = std::get<2>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data1                                 = std::get<3>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data2                                 = std::get<4>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data3                                 = std::get<5>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(M.as_int32(), 1, 1);
  dim3 block(N.as_int32(), 1, 1);
  cuda_module.LaunchKernel(0, "elementwise_add", grid, block, args);

  CUDA_CALL(cudaMemcpy(host_data3.data(),
                       reinterpret_cast<void*>(Cd),
                       M.as_int32() * N.as_int32() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < M.as_int32(); i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      int offset = i * N.as_int32() + j;
      EXPECT_NEAR(host_data3[offset], host_data1[offset] * host_data2[offset], 1e-5);
    }
  }
}

TEST(CodeGenCUDA3, test_of_mul_cachewrite) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr K(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A1", {M, K});
  Placeholder<float> B("B1", {N, K});

  auto k1 = Var(K.as_int32(), "k1");
  auto C  = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(j, k1), {k1}); }, "C1");

  auto stages = CreateStages({C});

  auto C_WC = stages[C]->CacheWrite("local", stages, C);
  stages[C]->Split(1, 2);
  stages[C]->Split(0, 4);
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[C_WC]->ComputeAt(stages[C], 2);

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("mul_cache_write", stages, {A, B, C}, {}, {}, nullptr, target);

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled CacheWrite and Reduce code:\n\n\n" << source_code;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void mul_cache_write(const float* __restrict__ A1, const float* __restrict__ B1, float* __restrict__ C1)
{
  float _C1_write_cache [ 2 ];
  float* C1_write_cache = _C1_write_cache;
  float* C1_write_cache__reduce_init = _C1_write_cache;
  if ((blockIdx.x < 8)) {
    if ((threadIdx.x < 4)) {
      for (int32_t j_outer = 0; j_outer < 16; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
          C1_write_cache__reduce_init[j_inner] = 0;
          for (int32_t k1 = 0; k1 < 32; k1 += 1) {
            C1_write_cache[j_inner] = (C1_write_cache[j_inner] + (A1[((128 * blockIdx.x) + ((32 * threadIdx.x) + k1))] * B1[((32 * j_inner) + ((64 * j_outer) + k1))]));
          };
        };
        for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
          C1[((128 * blockIdx.x) + ((2 * j_outer) + ((32 * threadIdx.x) + j_inner)))] = C1_write_cache[j_inner];
        };
      };
    };
  };
}

}
)ROC";
  ASSERT_EQ(utils::Trim(source_target), source_code);

  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  auto _Ad_Bd_Cd_host_data1_host_data2_host_data3_ = CreateNVMemory(M.as_int32(), N.as_int32());
  auto& Ad                                         = std::get<0>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Bd                                         = std::get<1>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Cd                                         = std::get<2>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data1                                 = std::get<3>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data2                                 = std::get<4>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data3                                 = std::get<5>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(8, 1, 1);
  dim3 block(4, 1, 1);
  cuda_module.LaunchKernel(0, "mul_cache_write", grid, block, args);

  CUDA_CALL(cudaMemcpy(host_data3.data(),
                       reinterpret_cast<void*>(Cd),
                       M.as_int32() * N.as_int32() * sizeof(float),
                       cudaMemcpyDeviceToHost));
  std::vector<float> res(1024, 0.0);
  for (int i = 0; i < M.as_int32(); i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      for (int k = 0; k < N.as_int32(); k++) {
        res[i * 32 + j] += host_data1[i * 32 + k] * host_data2[j * 32 + k];
      }
      int offset = i * 32 + j;
      EXPECT_NEAR(host_data3[offset], res[offset], 1e-3);
    }
  }
}

class ElementwiseTester {
 public:
  Expr N{212};
  Var M{"M"};

  explicit ElementwiseTester(const std::string& fn_name) : fn_name_(fn_name) {}

  std::tuple<Placeholder<float>, Placeholder<float>, ir::Tensor> BuildNet() {
    Target target = common::DefaultNVGPUTarget();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

    return std::make_tuple(A, B, C);
  }

  void Test(Placeholder<float>& A,  // NOLINT
            Placeholder<float>& B,  // NOLINT
            ir::Tensor& C,          // NOLINT
            std::vector<int> grid_sizes,
            std::vector<int> block_sizes) {
    Var M("M");
    auto stages = CreateStages({A, B, C});
    auto func   = Lower(fn_name_, stages, {A, B, C}, {M});
    LOG(INFO) << "func:\n" << func;

    Target target = common::DefaultNVGPUTarget();
    Module::Builder builder("module", target);
    builder.AddFunction(func);

    CodeGenCUDA_Dev codegen(target);
    auto source_code = codegen.Compile(builder.Build());

    LOG(INFO) << "compiled code:\n\n\n" << source_code;

    // compile the code
    using runtime::cuda::CUDAModule;

    backends::NVRTC_Compiler compiler;

    auto ptx = compiler(source_code);
    CHECK(!ptx.empty());

    CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

    // launch the kernel

    const int m            = N.as_int32();
    const int num_elements = m * N.as_int32();
    const int bytes        = num_elements * sizeof(float);

    CUdeviceptr Ad, Bd, Cd;
    cuMemAlloc(&Ad, bytes);
    cuMemAlloc(&Bd, bytes);
    cuMemAlloc(&Cd, bytes);

    std::vector<float> host_data1(num_elements, 0);
    std::vector<float> host_data2(num_elements, 0);
    std::vector<float> host_data3(num_elements, 0);
    for (int i = 0; i < num_elements; i++) {
      host_data1[i] = (rand() * 1.f) / INT_MAX;  // NOLINT
      host_data2[i] = (rand() * 1.f) / INT_MAX;  // NOLINT
    }

    CUDA_CALL(cudaMemcpy(
        reinterpret_cast<void*>(Ad), host_data1.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(
        reinterpret_cast<void*>(Bd), host_data2.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    void* args[] = {const_cast<int*>(&m), &Ad, &Bd, &Cd};

    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);
    if (grid_sizes.size() >= 1) {
      grid.x = grid_sizes[0];
    }
    if (grid_sizes.size() >= 2) {
      grid.y = grid_sizes[1];
    }
    if (grid_sizes.size() >= 3) {
      grid.z = grid_sizes[2];
    }
    if (block_sizes.size() >= 1) {
      block.x = block_sizes[0];
    }
    if (block_sizes.size() >= 2) {
      block.x = block_sizes[1];
    }
    if (block_sizes.size() >= 3) {
      block.x = block_sizes[2];
    }

    cuda_module.LaunchKernel(0, fn_name_, grid, block, args);

    CUDA_CALL(cudaMemcpy(
        host_data3.data(), reinterpret_cast<void*>(Cd), num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < N.as_int32(); j++) {
        int offset = i * N.as_int32() + j;
        if (i == 0 && j < 10) {
          LOG(INFO) << host_data3[offset];
        }
        EXPECT_NEAR(host_data3[offset], host_data1[offset] * host_data2[offset], 1e-5);
      }
    }

    CUDA_CALL(cudaFree(reinterpret_cast<void*>(Ad)))
    CUDA_CALL(cudaFree(reinterpret_cast<void*>(Bd)))
    CUDA_CALL(cudaFree(reinterpret_cast<void*>(Cd)))
  }

 private:
  std::string fn_name_;
};

TEST(CodeGenCUDA, jit_dynamic_shape0) {
  ElementwiseTester tester("elementwise_base");
  auto _A_B_C_ = tester.BuildNet();  // NOLINT
  auto& A      = std::get<0>(_A_B_C_);
  auto& B      = std::get<1>(_A_B_C_);
  auto& C      = std::get<2>(_A_B_C_);

  auto stages = CreateStages({C});

  auto _M_outer_M_inner_ = stages[C]->Split(0, 32);  // M/32, 32 NOLINT
  auto& M_outer          = std::get<0>(_M_outer_M_inner_);
  auto& M_inner          = std::get<1>(_M_outer_M_inner_);
  stages[C]->Reorder({
      M_inner,
      stages[C]->axis(2),
      M_outer,
  });

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  tester.Test(A, B, C, {32}, {tester.N.as_int32()});
}

TEST(CodeGenCUDA, jit_dynamic_shape1) {
  ElementwiseTester tester("elementwise1");
  auto _A_B_C_ = tester.BuildNet();  // NOLINT
  auto& A      = std::get<0>(_A_B_C_);
  auto& B      = std::get<1>(_A_B_C_);
  auto& C      = std::get<2>(_A_B_C_);

  auto stages = CreateStages({C});

  auto _M_outer_M_inner_ = stages[C]->Split(0, 32);  // M/32, 32 NOLINT
  auto& M_outer          = std::get<0>(_M_outer_M_inner_);
  auto& M_inner          = std::get<1>(_M_outer_M_inner_);
  auto _N_outer_N_inner_ = stages[C]->Split(2, 32);  // M/32, 32 NOLINT
  auto& N_outer          = std::get<0>(_N_outer_N_inner_);
  auto& N_inner          = std::get<1>(_N_outer_N_inner_);
  stages[C]->Reorder({
      M_inner,
      N_inner,
      M_outer,
      N_outer,
  });

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  tester.Test(A, B, C, {32}, {32});
}

TEST(CodeGenCUDA, jit_dynamic_shape2) {
  ElementwiseTester tester("elementwise2");

  auto _A_B_C_ = tester.BuildNet();  // NOLINT
  auto& A      = std::get<0>(_A_B_C_);
  auto& B      = std::get<1>(_A_B_C_);
  auto& C      = std::get<2>(_A_B_C_);

  auto stages = CreateStages({C});

  auto _M_outer_M_inner_ = stages[C]->Split(0, 32);  // M/32, 32 NOLINT
  auto& M_outer          = std::get<0>(_M_outer_M_inner_);
  auto& M_inner          = std::get<1>(_M_outer_M_inner_);
  auto _N_outer_N_inner_ = stages[C]->Split(2, 3);  // M/32, 32 NOLINT
  auto& N_outer          = std::get<0>(_N_outer_N_inner_);
  auto& N_inner          = std::get<1>(_N_outer_N_inner_);
  stages[C]->Reorder({
      M_inner,
      N_inner,
      M_outer,
      N_outer,
  });

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  tester.Test(A, B, C, {32}, {3});
}

TEST(CodeGenCUDA, jit_host_call_cuda_kernel) {
  auto _Ad_Bd_Cd_host_data1_host_data2_host_data3_ = CreateNVMemory(100, 200);
  auto& Ad                                         = std::get<0>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Bd                                         = std::get<1>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& Cd                                         = std::get<2>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data1                                 = std::get<3>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data2                                 = std::get<4>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);
  auto& host_data3                                 = std::get<5>(_Ad_Bd_Cd_host_data1_host_data2_host_data3_);

  ElementwiseTester tester("elementwise_host_test");
  auto _A_B_C_ = tester.BuildNet();  // NOLINT
  auto& A      = std::get<0>(_A_B_C_);
  auto& B      = std::get<1>(_A_B_C_);
  auto& C      = std::get<2>(_A_B_C_);
  auto stages  = CreateStages({C});

  auto _M_outer_M_inner_ = stages[C]->Split(0, 32);  // M/32, 32 NOLINT
  auto& M_outer          = std::get<0>(_M_outer_M_inner_);
  auto& M_inner          = std::get<1>(_M_outer_M_inner_);
  auto _N_outer_N_inner_ = stages[C]->Split(2, 3);  // M/32, 32 NOLINT
  auto& N_outer          = std::get<0>(_N_outer_N_inner_);
  auto& N_inner          = std::get<1>(_N_outer_N_inner_);
  stages[C]->Reorder({
      M_inner,
      N_inner,
      M_outer,
      N_outer,
  });

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  Var M("M");
  auto func = Lower("fn", stages, {A, B, C}, {M});

  LOG(INFO) << "func:\n" << func;

  Target target = common::DefaultNVGPUTarget();
  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto module = builder.Build();
  Expr expr(module);

  auto _host_module_device_module_ = SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module                = std::get<0>(_host_module_device_module_);
  auto& device_module              = std::get<1>(_host_module_device_module_);
  for (auto& func : host_module.functions()) {
    LOG(INFO) << "host:\n" << func;
  }

  for (auto& func : device_module.functions()) {
    LOG(INFO) << "device:\n" << func;
  }

  void* fn_kernel;
  void* stream = nullptr;

  // compile with device
  CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);
  fn_kernel = cuda_module.GetFunction(0, "fn");
  CHECK(fn_kernel);

  LOG(INFO) << "fn_kernel: " << fn_kernel;

  RuntimeSymbolRegistry::Global().RegisterFn("fn_kernel_ptr_", reinterpret_cast<void*>(&fn_kernel));
  RuntimeSymbolRegistry::Global().RegisterVar("fn_kernel_stream_ptr_", stream);

  // compile host
  {
    auto jit = SimpleJIT::Create();
    jit->Link<CodeGenCUDA_Host>(host_module);

    auto fn_ptr = jit->Lookup("fn");
    CHECK(fn_ptr);

    Expr M(100);
    Expr N(200);

    cinn_buffer_t* A_buf =
        cinn_buffer_new(cinn_x86_device, cinn_float32_t(), std::vector<int>{{M.as_int32(), N.as_int32()}});
    cinn_buffer_t* B_buf =
        cinn_buffer_new(cinn_x86_device, cinn_float32_t(), std::vector<int>{{M.as_int32(), N.as_int32()}});
    cinn_buffer_t* C_buf =
        cinn_buffer_new(cinn_x86_device, cinn_float32_t(), std::vector<int>{{M.as_int32(), N.as_int32()}});

    A_buf->memory = reinterpret_cast<uint8_t*>(Ad);
    B_buf->memory = reinterpret_cast<uint8_t*>(Bd);
    C_buf->memory = reinterpret_cast<uint8_t*>(Cd);

    CUDA_CALL(cudaDeviceSynchronize());

    // call the kernel
    auto comp = reinterpret_cast<void (*)(cinn_pod_value_t*, int)>(fn_ptr);

    auto args = common::ArgsBuilder().Add(M.as_int32()).Add(A_buf).Add(B_buf).Add(C_buf).Build();

    comp(args.data(), args.size());

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(host_data3.data(),
                         reinterpret_cast<void*>(Cd),
                         M.as_int32() * N.as_int32() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < M.as_int32(); i++) {
      for (int j = 0; j < N.as_int32(); j++) {
        int offset = i * N.as_int32() + j;
        if (i == 0 && j < 4) {
          LOG(INFO) << host_data3[offset];
        }
        ASSERT_NEAR(host_data3[offset], host_data1[offset] * host_data2[offset], 1e-5);
      }
    }
  }
}

TEST(depthwise_conv, test) {
  const int batch       = 4;
  const int in_channel  = 3;
  const int in_height   = 40;
  const int in_width    = 40;
  const int filter_size = 4;

  const int pad_left   = 3;
  const int pad_right  = 3;
  const int pad_top    = 3;
  const int pad_bottom = 3;
  const int stride     = 1;

  const int height_padded = in_height + pad_top + pad_bottom;
  const int width_padded  = in_width + pad_left + pad_right;

  const int out_channel = in_channel;
  const int out_height  = height_padded - filter_size;
  const int out_width   = width_padded - filter_size;

  Placeholder<float> input("input", {Expr(batch), Expr(in_channel), Expr(in_height), Expr(in_width)});
  Placeholder<float> filter("filter", {Expr(in_channel), Expr(in_channel), Expr(filter_size), Expr(filter_size)});

  auto padded_input = Compute(
      {Expr(batch), Expr(in_channel), Expr(height_padded), Expr(width_padded)},
      [=](Expr b, Expr c, Expr i, Expr j) {
        return common::select(common::and_all({
                                  i >= pad_top,
                                  i - pad_top < in_height,
                                  j >= pad_left,
                                  j - pad_left < in_width,
                              }),
                              input(b, c, i, j),  // true value
                              Expr(0.f)           // false value
        );                                        // NOLINT
      },
      "padded_input");

  Var di(Expr(filter_size), "di");
  Var dj(Expr(filter_size), "dj");

  // cache
  auto IS = Compute(
      padded_input->shape,
      [=](Expr b, Expr c, Expr i, Expr j) -> Expr { return padded_input(b, c, i, j); },
      "cache_paded_input");
  auto FS = Compute(
      ir::Tensor(filter)->shape,
      [=](Expr c0, Expr c1, Expr w, Expr h) -> Expr { return filter(c0, c1, w, h); },
      "cache_filter");

  auto output = Compute(
      {Expr(batch), Expr(in_channel), Expr(out_height), Expr(out_width)},
      [=](Var b, Var c, Var i, Var j) -> Expr {
        auto expr = IS(b, c, i * stride + di, j * stride + dj) * FS(c, c, di, dj);
        return lang::ReduceSum(expr, {di, dj});
      },
      "output");

  auto stages = CreateStages({output});

  stages[padded_input]->ComputeInline();

  stages[output]->Fuse(0, 1);
  stages[output]->Fuse(0, 1);
  stages[output]->Split(0, 20);

  auto fn = Lower("fn", stages, {input, filter, IS, FS, output});

  LOG(INFO) << "fn:\n" << fn;
}

TEST(Conv, basic) {
  Expr batch(256);
  Expr in_channel(256);
  Expr out_channel(512);
  Expr in_size(14);
  Expr pad(1);
  Expr kernel(3);
  Expr stride(1);
  Expr out_size = (in_size - kernel + 2 * pad) / stride + 1;

  Placeholder<float> A("A", {in_size, in_size, in_channel, batch});
  Placeholder<float> W("W", {kernel, kernel, in_channel, out_channel});

  auto Apad = Compute(
      {in_size + 2 * pad, in_size + 2 * pad, in_channel, batch},
      [=](Expr yy, Expr xx, Expr cc, Expr nn) -> Expr {
        return common::select(common::and_all({yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size}),
                              A(yy - pad, xx - pad, cc, nn),
                              common::make_const(Float(32), 0));
      },
      "Apad");

  Var rc(in_channel, "rc");
  Var ry(kernel, "ry");
  Var rx(kernel, "rx");

  auto B = Compute(
      {out_size, out_size, out_channel, batch},
      [=](Expr yy, Expr xx, Expr ff, Expr nn) -> Expr {
        return lang::ReduceSum(Apad(yy * stride + ry, xx * stride + rx, rc, nn) * W(ry, rx, rc, ff), {rc, ry, rx});
      },
      "B");

  auto stages = CreateStages({A, W, Apad, B});
  stages[Apad]->ComputeInline();

  auto fn = Lower("Conv2d_basic", stages, {A, W, B});

  LOG(INFO) << "Conv2d_basic:\n" << fn;
}

TEST(elementwise_add1, share_local_cache) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(200);
  Expr K(300);
  Expr P(400);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  auto stages = CreateStages({C});

  std::vector<ir::Tensor> temp{C};
  auto AA = stages[A]->CacheRead("local", temp, stages);
  auto AL = stages[AA]->CacheRead("local", temp, stages);
  // NOTE here, the CC replace the C as the output the function.
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[AL]->ComputeAt(stages[C], 1);
  stages[AA]->ComputeAt(stages[AL], 1);

  Module::Builder builder("gpu_module", common::DefaultNVGPUTarget());

  auto fn = Lower("elementwise_add", stages, {A, B, C});

  builder.AddFunction(fn);
  auto module = builder.Build();

  // compile with device code
  CodeGenCUDA_Dev codegen(common::DefaultNVGPUTarget());
  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "source code of share_local_cache is: " << source_code;
  backends::NVRTC_Compiler compiler;

  common::CudaModuleTester tester;
  tester.Compile(module);

  auto* A_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* B_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* C_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();
  auto* C_target_host = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* B_dev = tester.CreateDeviceBuffer(B_host);
  auto* C_dev = tester.CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[3];
  for (int i = 0; i < 3; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args           = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("elementwise_add", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->memory);
  auto* B_mem        = reinterpret_cast<float*>(B_host->memory);
  for (int i = 0; i < C_target_host->num_elements(); i++) {
    if ((C_target_mem[i] - A_mem[i] - B_mem[i]) > 0.0001 || (C_target_mem[i] - A_mem[i] - B_mem[i]) < -0.0001) {
      LOG(INFO) << "The target should be: " << C_target_mem[i] << ", but result is: " << A_mem[i] + B_mem[i];
    }
    ASSERT_NEAR(C_target_mem[i], A_mem[i] + B_mem[i], 1e-3);
  }

  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(B_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}

TEST(elementwise_add0, share_local_cache) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(20);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  auto stages = CreateStages({C});

  auto CC = stages[C]->CacheWrite("local", stages, C);
  std::vector<ir::Tensor> temp{CC};
  auto AA = stages[A]->CacheRead("shared", temp, stages);
  // NOTE here, the CC replace the C as the output the function.

  stages[CC]->ComputeAt(stages[C], 1);
  stages[AA]->ComputeAt(stages[CC], 1);
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  Module::Builder builder("gpu_module", common::DefaultNVGPUTarget());

  auto fn = Lower("elementwise_add0", stages, {A, B, C}, {}, {AA, CC});

  ASSERT_EQ(fn->temp_bufs.size(), 2UL);
  builder.AddFunction(fn);
  auto module = builder.Build();

  auto _host_module_device_module_ = SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module                = std::get<0>(_host_module_device_module_);
  auto& device_module              = std::get<1>(_host_module_device_module_);
  for (auto& func : host_module.functions()) {
    LOG(INFO) << "host:\n" << func;
  }

  for (auto& func : device_module.functions()) {
    LOG(INFO) << "device:\n" << func;
  }

  // compile with device code
  CodeGenCUDA_Dev codegen(common::DefaultNVGPUTarget());
  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "device source code elementwise_add0:\n" << source_code;

  backends::NVRTC_Compiler compiler;

  common::CudaModuleTester tester;
  tester.Compile(module);

  auto* A_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* B_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* C_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();
  auto* C_target_host = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* B_dev = tester.CreateDeviceBuffer(B_host);
  auto* C_dev = tester.CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[3];
  for (int i = 0; i < 3; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args           = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("elementwise_add0", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->memory);
  auto* B_mem        = reinterpret_cast<float*>(B_host->memory);
  for (int i = 0; i < C_target_host->num_elements(); i++) {
    ASSERT_NEAR(C_target_mem[i], A_mem[i] + B_mem[i], 1e-5);
  }

  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(B_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}

TEST(Conv, optimize) {
  Context::Global().ResetNameId();
  // basic implementation
  Expr batch(256);
  Expr in_channel(256);
  Expr out_channel(512);
  Expr in_size(14);
  Expr kernel(3);
  Expr pad(1);
  Expr stride(1);

  auto A = Placeholder<float>("A", {in_size, in_size, in_channel, batch});
  auto W = Placeholder<float>("W", {kernel, kernel, in_channel, out_channel});

  Expr out_size((in_size.as_int32() - kernel.as_int32() + 2 * pad.as_int32()) / stride.as_int32() + 1);

  auto Apad = Compute(
      {in_size + 2 * pad, in_size + 2 * pad, in_channel, batch},
      [&](Expr yy, Expr xx, Expr cc, Expr nn) {
        auto condition = common::and_all({yy >= pad, xx - pad < in_size, xx >= pad, xx - pad < in_size});
        return common::select(condition, A(yy - pad, xx - pad, cc, nn), common::make_const(0.f));
      },
      "Apad");

  auto rc = Var(in_channel, "rc");
  auto ry = Var(kernel, "ry");
  auto rx = Var(kernel, "rx");

  auto B = Compute(
      {out_size, out_size, out_channel, batch},
      [=](Expr yy, Expr xx, Expr ff, Expr nn) {
        return lang::ReduceSum(Apad(yy * stride + ry, xx * stride + rx, rc, nn) * W(ry, rx, rc, ff), {rc, ry, rx});
      },
      "B");

  auto stages = CreateStages({B});
  std::vector<ir::Tensor> temp{B};
  auto BL = stages[B]->CacheWrite("local", stages, B);
  auto AA = stages[Apad]->CacheRead("shared", temp, stages);
  auto WW = stages[W]->CacheRead("shared", temp, stages);
  auto AL = stages[AA]->CacheRead("local", temp, stages);
  auto WL = stages[WW]->CacheRead("local", temp, stages);

  stages[Apad]->ComputeInline();

  // tile consts
  const int tile         = 8;
  const int num_thread   = 8;
  const int block_factor = tile * num_thread;
  const int step         = 8;
  const int vthread      = 2;

  auto hi = stages[BL]->axis(0);
  auto wi = stages[BL]->axis(1);
  auto fi = stages[BL]->axis(2);
  auto ni = stages[BL]->axis(3);
  auto bz = stages[BL]->Fuse(hi, wi);

  poly::Iterator by, bx, ty, tx;
  std::tie(by, fi) = stages[BL]->Split(fi, block_factor);  // NOLINT
  std::tie(bx, ni) = stages[BL]->Split(ni, block_factor);  // NOLINT

  poly::Iterator tyz, txz;
  std::tie(tyz, fi) = stages[BL]->Split(fi, vthread);
  std::tie(txz, ni) = stages[BL]->Split(ni, vthread);
  std::tie(ty, fi)  = stages[BL]->Split(fi, num_thread);
  std::tie(tx, ni)  = stages[BL]->Split(ni, num_thread);
  stages[BL]->Reorder({bz, by, bx, tyz, txz, ty, tx, fi, ni});

  LOG(INFO) << "Conv.optimize function is:\n" << Lower("conv", stages, {A, W, B}, {}, {AA, WW, AL, WL, BL});
}

TEST(ElementwiseAdd, cache_read_local) {
  Context::Global().ResetNameId();

  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  auto stages = CreateStages({C});

  std::vector<ir::Tensor> temp{C};

  auto AL = stages[A]->CacheRead("local", temp, stages);
  stages[C]->Split(0, 10);
  stages[AL]->ComputeAt(stages[C], 1);
  stages[C]->Bind(0, "threadIdx.x");
  stages[C]->Bind(1, "blockIdx.x");

  Target target = common::DefaultNVGPUTarget();
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn0", stages, {A, B, C}, {}, {AL});

  Module::Builder builder("module", target);
  builder.AddFunction(fn);

  auto module      = builder.Build();
  auto source_code = codegen.Compile(module);
  LOG(INFO) << "source cache_read_local:\n" << source_code;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn0(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  float _A_read_cache [ 200 ];
  float* A_read_cache = _A_read_cache;
  if ((threadIdx.x < 10)) {
    if ((blockIdx.x < 10)) {
    {
      for (int32_t j = 0; j < 200; j += 1) {
        A_read_cache[j] = A[((200 * blockIdx.x) + ((2000 * threadIdx.x) + j))];
      };
      for (int32_t j = 0; j < 200; j += 1) {
        C[((200 * blockIdx.x) + ((2000 * threadIdx.x) + j))] = (A_read_cache[j] + B[((200 * blockIdx.x) + ((2000 * threadIdx.x) + j))]);
      };
    }
    };
  };
}

}
)ROC";
  ASSERT_EQ(utils::Trim(source_target), source_code);

  backends::NVRTC_Compiler compiler;

  common::CudaModuleTester tester;
  tester.Compile(module);

  auto* A_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* B_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* C_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();
  auto* C_target_host = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* B_dev = tester.CreateDeviceBuffer(B_host);
  auto* C_dev = tester.CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[3];
  for (int i = 0; i < 3; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args           = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("fn0", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->memory);
  auto* B_mem        = reinterpret_cast<float*>(B_host->memory);
  for (int i = 0; i < C_target_host->num_elements(); i++) {
    ASSERT_NEAR(C_target_mem[i], A_mem[i] + B_mem[i], 1e-5);
  }

  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(B_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}

TEST(ElementwiseAdd, cache_read1) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(200);

  auto create_module = [&] {
    Context::Global().ResetNameId();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M - 2, N}, [&](Expr i, Expr j) { return A(i, j) + A(i + 1, j) + A(i + 2, j) + B(i, j); }, "C");

    std::vector<ir::Tensor> temp{C};
    auto stages = CreateStages(temp);
    auto AL     = stages[A]->CacheRead("local", temp, stages);
    stages[C]->Split(1, 10);
    stages[AL]->ComputeAt2(stages[C], 2);

    return std::make_tuple(A, B, C, AL, stages);
  };
  {
    auto _A_B_C_AL_stages_ = create_module();  // NOLINT
    auto& A                = std::get<0>(_A_B_C_AL_stages_);
    auto& B                = std::get<1>(_A_B_C_AL_stages_);
    auto& C                = std::get<2>(_A_B_C_AL_stages_);
    auto& AL               = std::get<3>(_A_B_C_AL_stages_);
    auto& stages           = std::get<4>(_A_B_C_AL_stages_);
    auto fn                = Lower("fn1", stages, {A, B, C}, {}, {AL});
    CodeGenC codegen_c(common::DefaultHostTarget());
    codegen_c.SetInlineBuiltinCodes(false);

    Module::Builder builder("module", common::DefaultHostTarget());
    builder.AddFunction(fn);

    auto c_source_code = codegen_c.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
    std::cout << "C source code:\n" << c_source_code << std::endl;
  }

  auto _A_B_C_AL_stages_ = create_module();  // NOLINT
  auto& A                = std::get<0>(_A_B_C_AL_stages_);
  auto& B                = std::get<1>(_A_B_C_AL_stages_);
  auto& C                = std::get<2>(_A_B_C_AL_stages_);
  auto& AL               = std::get<3>(_A_B_C_AL_stages_);
  auto& stages           = std::get<4>(_A_B_C_AL_stages_);
  stages[C]->Bind(0, "threadIdx.x");
  stages[C]->Bind(1, "blockIdx.x");

  Target target = common::DefaultNVGPUTarget();
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn1", stages, {A, B, C}, {}, {AL});
  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  std::cout << "CUDA source of ComputeAt2 & CacheRead\n" << source_code << std::endl;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn1(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  float _A_read_cache [ 3 ];
  float* A_read_cache = _A_read_cache;
  if ((threadIdx.x < 98)) {
    if ((blockIdx.x < 20)) {
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        for (int32_t i_at = 0; i_at < 3; i_at += 1) {
          A_read_cache[i_at] = A[((10 * blockIdx.x) + ((200 * i_at) + ((200 * threadIdx.x) + j_inner)))];
        };
        C[((10 * blockIdx.x) + ((200 * threadIdx.x) + j_inner))] = (A_read_cache[0] + (A_read_cache[1] + (A_read_cache[2] + B[((10 * blockIdx.x) + ((200 * threadIdx.x) + j_inner))])));
      };
    };
  };
}

}
)ROC";

  ASSERT_EQ(utils::Trim(source_target), source_code);

  common::CudaModuleTester tester;
  tester.Compile(builder.Build());

  auto* A_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* B_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* C_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();
  auto* C_target_host = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* B_dev = tester.CreateDeviceBuffer(B_host);
  auto* C_dev = tester.CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[3];
  for (int i = 0; i < 3; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args           = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("fn1", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->memory);
  auto* B_mem        = reinterpret_cast<float*>(B_host->memory);
  for (int i = 0; i < M.as_int32() - 2; i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      ASSERT_NEAR(C_target_mem[i * N.as_int32() + j],
                  A_mem[i * N.as_int32() + j] + A_mem[(i + 1) * N.as_int32() + j] + A_mem[(i + 2) * N.as_int32() + j] +
                      B_mem[i * N.as_int32() + j],
                  1e-4);
    }
  }

  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(B_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}

TEST(ElementwiseAdd, cache_read_compute_at1) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(95);
  Context::Global().ResetNameId();
  Placeholder<float> A("AA", {M, M});

  auto C = Compute(
      {N, N}, [&](Expr i, Expr j) { return A(i, j) + A(i + 2, j + 2) + A(i + 5, j + 5); }, "C");

  auto stages = CreateStages({A, C});
  std::vector<ir::Tensor> temp{C};
  auto AL = stages[A]->CacheRead("shared", temp, stages);
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[AL]->ComputeAt2(stages[C], 1);

  Target target = common::DefaultNVGPUTarget();
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn_cacheread_computeat1", stages, {A, C});
  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "CUDA source of cache_read_compute_at1:\n" << source_code << std::endl;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn_cacheread_computeat1(const float* __restrict__ AA, float* __restrict__ C)
{
  __shared__ float _AA_read_cache [ 600 ];
  float* AA_read_cache = _AA_read_cache;
  if ((blockIdx.x < 95)) {
    if ((threadIdx.x < 95)) {
    {
      for (int32_t i_at = 0; i_at < 6; i_at += 1) {
        for (int32_t j_at = 0; j_at < 6; j_at += 1) {
          AA_read_cache[((100 * i_at) + (j_at + threadIdx.x))] = AA[((100 * blockIdx.x) + ((100 * i_at) + (j_at + threadIdx.x)))];
        };
      };
      C[((95 * blockIdx.x) + threadIdx.x)] = (AA_read_cache[threadIdx.x] + (AA_read_cache[(202 + threadIdx.x)] + AA_read_cache[(505 + threadIdx.x)]));
    }
    };
  };
}

}
)ROC";

  ASSERT_EQ(utils::Trim(source_target), source_code);

  common::CudaModuleTester tester;
  tester.Compile(builder.Build());

  auto* A_host        = common::BufferBuilder(Float(32), {M.as_int32(), M.as_int32()}).set_random().Build();
  auto* C_host        = common::BufferBuilder(Float(32), {N.as_int32(), N.as_int32()}).set_zero().Build();
  auto* C_target_host = common::BufferBuilder(Float(32), {N.as_int32(), N.as_int32()}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* C_dev = tester.CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[2];
  for (int i = 0; i < 2; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args           = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("fn_cacheread_computeat1", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->memory);
  for (int i = 0; i < N.as_int32(); i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      ASSERT_NEAR(
          C_target_mem[i * N.as_int32() + j],
          (A_mem[i * M.as_int32() + j] + A_mem[(i + 2) * M.as_int32() + j + 2] + A_mem[(i + 5) * M.as_int32() + j + 5]),
          1e-4);
    }
  }
  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}

TEST(ElementwiseAdd, cache_read_compute_at2) {
  Expr M(100);
  Expr N(50);
  Context::Global().ResetNameId();
  Placeholder<float> A("AA", {M, M});

  auto C = Compute(
      {N, N}, [&](Expr i, Expr j) { return A(i + 5, j) + A(i, j + 5); }, "C");

  auto stages = CreateStages({A, C});
  std::vector<ir::Tensor> temp{C};
  auto AL = stages[A]->CacheRead("local", temp, stages);
  stages[C]->Split(1, 10);
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[AL]->ComputeAt2(stages[C], 2);

  Target target = common::DefaultNVGPUTarget();
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn_cacheread_computeat2", stages, {A, C});
  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "CUDA source of cache_read_compute_at2:\n" << source_code << std::endl;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn_cacheread_computeat2(const float* __restrict__ AA, float* __restrict__ C)
{
  float _AA_read_cache [ 36 ];
  float* AA_read_cache = _AA_read_cache;
  if ((blockIdx.x < 50)) {
    if ((threadIdx.x < 5)) {
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        for (int32_t i_at = 0; i_at < 6; i_at += 1) {
          for (int32_t j_at = 0; j_at < 6; j_at += 1) {
            AA_read_cache[((6 * i_at) + j_at)] = AA[((100 * blockIdx.x) + ((100 * i_at) + ((10 * threadIdx.x) + (j_at + j_inner))))];
          };
        };
        C[((50 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))] = (AA_read_cache[30] + AA_read_cache[5]);
      };
    };
  };
}

}
)ROC";

  ASSERT_EQ(utils::Trim(source_target), source_code);

  common::CudaModuleTester tester;
  tester.Compile(builder.Build());

  auto* A_host        = common::BufferBuilder(Float(32), {M.as_int32(), M.as_int32()}).set_random().Build();
  auto* C_host        = common::BufferBuilder(Float(32), {N.as_int32(), N.as_int32()}).set_zero().Build();
  auto* C_target_host = common::BufferBuilder(Float(32), {N.as_int32(), N.as_int32()}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* C_dev = tester.CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[2];
  for (int i = 0; i < 2; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args           = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("fn_cacheread_computeat2", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->memory);
  for (int i = 0; i < N.as_int32(); i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      ASSERT_NEAR(C_target_mem[i * N.as_int32() + j],
                  (A_mem[(i + 5) * M.as_int32() + j] + A_mem[i * M.as_int32() + j + 5]),
                  1e-4);
    }
  }
  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}

// JIT test precision for the basic elementwise add
void TestElementwiseAddPrecisionBasic(
    const ir::Module& module,
    const std::string& fn_name,
    Expr M,
    Expr N,
    std::function<float(float, float)> elem_cal = [](float a, float b) { return a + b; }) {
  common::CudaModuleTester tester;
  tester.Compile(module);

  auto* A_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* B_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* C_host        = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();
  auto* C_target_host = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* B_dev = tester.CreateDeviceBuffer(B_host);
  auto* C_dev = tester.CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[3];
  for (int i = 0; i < 3; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args           = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester(fn_name, args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->memory);
  auto* B_mem        = reinterpret_cast<float*>(B_host->memory);
  for (int i = 0; i < M.as_int32(); i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      ASSERT_NEAR(
          C_target_mem[i * N.as_int32() + j], elem_cal(A_mem[i * N.as_int32() + j], B_mem[i * N.as_int32() + j]), 1e-5);
    }
  }

  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(B_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}

TEST(ElementwiseAdd, cache_read_shared) {
  Context::Global().ResetNameId();

  Expr M(100);
  Expr N(200);

  auto create_module = [&] {
    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");
    auto stages = CreateStages({A, B, C});
    std::vector<ir::Tensor> temp{C};
    auto AL = stages[A]->CacheRead("shared", temp, stages);

    stages[C]->Bind(0, "blockIdx.x");
    stages[AL]->ComputeAt(stages[C], 1);

    return std::make_tuple(A, B, C, AL, stages);
  };

  auto _A_B_C_AL_stages_ = create_module();  // NOLINT
  auto& A                = std::get<0>(_A_B_C_AL_stages_);
  auto& B                = std::get<1>(_A_B_C_AL_stages_);
  auto& C                = std::get<2>(_A_B_C_AL_stages_);
  auto& AL               = std::get<3>(_A_B_C_AL_stages_);
  auto& stages           = std::get<4>(_A_B_C_AL_stages_);
  Target target          = common::DefaultNVGPUTarget();
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn2", stages, {A, B, C}, {}, {AL});

  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  std::cout << "CUDA source2:\n" << source_code << std::endl;

  auto target_source = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn2(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  __shared__ float _A_read_cache [ 1 ];
  float* A_read_cache = _A_read_cache;
  if ((blockIdx.x < 100)) {
    for (int32_t j = 0; j < 200; j += 1) {
      A_read_cache[0] = A[((200 * blockIdx.x) + j)];
      C[((200 * blockIdx.x) + j)] = (A_read_cache[0] + B[((200 * blockIdx.x) + j)]);
    };
  };
}

}
)ROC";

  LOG(INFO) << "GPU thread config: " << fn->cuda_axis_info;

  ASSERT_EQ(utils::Trim(target_source), source_code);

  TestElementwiseAddPrecisionBasic(builder.Build(), "fn2", M, N);
}

TEST(ElementwiseAdd, cache_write_local) {
  Context::Global().ResetNameId();
  // Make a small shape, because the shared memory is small.
  Expr M(40);
  Expr N(40);

  auto create_module = [&] {
    Context::Global().ResetNameId();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

    auto stages = CreateStages({A, B, C});

    auto Co = stages[C]->CacheWrite("local", stages, C);

    // Cache write local, the local memory can just share in a single thread, so it must ComputeAt(inside) the innermost
    // thread.
    stages[C]->Split(1, 4);
    stages[C]->Split(0, 4);
    stages[Co]->ComputeAt(stages[C], 1);
    stages[Co]->Split(2, 5);
    stages[C]->Bind(0, "blockIdx.x");
    stages[C]->Bind(1, "threadIdx.x");

    return std::make_tuple(A, B, C, Co, stages);
  };

  auto _A_B_C_Co_stages_ = create_module();  // NOLINT
  auto& A                = std::get<0>(_A_B_C_Co_stages_);
  auto& B                = std::get<1>(_A_B_C_Co_stages_);
  auto& C                = std::get<2>(_A_B_C_Co_stages_);
  auto& Co               = std::get<3>(_A_B_C_Co_stages_);
  auto& stages           = std::get<4>(_A_B_C_Co_stages_);

  CodeGenCUDA_Dev codegen(common::DefaultNVGPUTarget());

  auto fn = Lower("cache_write_local", stages, {A, B, C}, {}, {Co});

  Module::Builder builder("module", common::DefaultNVGPUTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  std::cout << "CUDA source cache_write:\n" << source_code << std::endl;

  auto target_source = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void cache_write_local(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  float _C_write_cache [ 40 ];
  float* C_write_cache = _C_write_cache;
  if ((blockIdx.x < 10)) {
    if ((threadIdx.x < 4)) {
    {
      for (int32_t j_outer = 0; j_outer < 8; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < 5; j_inner += 1) {
          C_write_cache[((5 * j_outer) + j_inner)] = (A[((160 * blockIdx.x) + ((5 * j_outer) + ((40 * threadIdx.x) + j_inner)))] + B[((160 * blockIdx.x) + ((5 * j_outer) + ((40 * threadIdx.x) + j_inner)))]);
        };
      };
      for (int32_t j_outer = 0; j_outer < 10; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < 4; j_inner += 1) {
          C[((160 * blockIdx.x) + ((4 * j_outer) + ((40 * threadIdx.x) + j_inner)))] = C_write_cache[((4 * j_outer) + j_inner)];
        };
      };
    }
    };
  };
}

}
)ROC";

  LOG(INFO) << "GPU thread config: " << fn->cuda_axis_info;

  ASSERT_EQ(utils::Trim(target_source), source_code);

  TestElementwiseAddPrecisionBasic(builder.Build(), "cache_write_local", M, N);
}

TEST(Cuda, external_function) {
  Context::Global().ResetNameId();
  // Make a small shape, because the shared memory is small.
  Expr M(40);
  Expr N(40);

  auto create_module = [&] {
    Context::Global().ResetNameId();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [&](Expr i, Expr j) { return CallExtern("tanh", {A(i, j)}) + CallExtern("cos", {B(i, j)}); }, "C");

    auto stages = CreateStages({A, B, C});

    stages[C]->Split(1, 10);
    stages[C]->Bind(0, "blockIdx.x");
    stages[C]->Bind(1, "threadIdx.x");

    return std::make_tuple(A, B, C, stages);
  };

  auto _A_B_C_stages_ = create_module();  // NOLINT
  auto& A             = std::get<0>(_A_B_C_stages_);
  auto& B             = std::get<1>(_A_B_C_stages_);
  auto& C             = std::get<2>(_A_B_C_stages_);
  auto& stages        = std::get<3>(_A_B_C_stages_);
  Target target       = common::DefaultNVGPUTarget();
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("external_function", stages, {A, B, C}, {}, {}, nullptr, target);

  Module::Builder builder("module", common::DefaultNVGPUTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  std::cout << "CUDA source:\n" << source_code << std::endl;

  auto target_source = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void external_function(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  if ((blockIdx.x < 40)) {
    if ((threadIdx.x < 4)) {
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        C[((40 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))] = (cinn_nvgpu_tanh_fp32(A[((40 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))]) + cinn_nvgpu_cos_fp32(B[((40 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))]));
      };
    };
  };
}

}
)ROC";

  LOG(INFO) << "GPU thread config: " << fn->cuda_axis_info;

  ASSERT_EQ(utils::Trim(target_source), source_code);

  TestElementwiseAddPrecisionBasic(
      builder.Build(), "external_function", M, N, [](float a, float b) { return std::tanh(a) + std::cos(b); });
}
#ifdef CINN_WITH_CUDNN
TEST(Cudnn, external_function_cudnn) {
  Context::Global().ResetNameId();

  common::CudaModuleTester tester;

  auto* A_host = common::BufferBuilder(Float(32), {2, 512, 7, 7}).set_random().Build();
  auto* B_host = common::BufferBuilder(Float(32), {512, 512, 3, 3}).set_random().Build();
  auto* C_host = common::BufferBuilder(Float(32), {2, 512, 7, 7}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* B_dev = tester.CreateDeviceBuffer(B_host);
  auto* C_dev = tester.CreateDeviceBuffer(C_host);

  cinn_buffer_t* dev_bufs[3];
  for (int i = 0; i < 3; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->memory = reinterpret_cast<uint8_t*>(C_dev);

  std::vector<int> attrs                          = {2, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 512, 7, 7};
  absl::flat_hash_map<std::string, int> attrs_map = {
      {"input_n", attrs[0]},     {"input_c", attrs[1]},     {"input_h", attrs[2]},   {"input_w", attrs[3]},
      {"weights_n", attrs[4]},   {"weights_c", attrs[5]},   {"weights_h", attrs[6]}, {"weights_w", attrs[7]},
      {"pad_h", attrs[8]},       {"pad_w", attrs[9]},       {"stride_h", attrs[10]}, {"stride_w", attrs[11]},
      {"dilation_h", attrs[12]}, {"dilation_w", attrs[13]}, {"groups", attrs[14]},   {"output_n", attrs[15]},
      {"output_c", attrs[16]},   {"output_h", attrs[17]},   {"output_w", attrs[18]},
  };

  runtime::cuda::cinn_gpu_cudnn_conv2d(

      attrs_map, dev_bufs[0], dev_bufs[1], dev_bufs[2]);
}

TEST(Cudnn, external_function_cudnn2) {
  Context::Global().ResetNameId();

  common::CudaModuleTester tester;

  auto* A_host = common::BufferBuilder(Float(32), {2, 64, 112, 112}).set_random().Build();
  auto* B_host = common::BufferBuilder(Float(32), {2, 64, 56, 56}).set_random().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* B_dev = tester.CreateDeviceBuffer(B_host);

  cinn_buffer_t* dev_bufs[2];
  for (int i = 0; i < 2; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);

  runtime::cuda::cinn_gpu_cudnn_pool2d(
      {2, 64, 112, 112, 3, 3, 1, 1, 1, 1, 2, 2, 2, 64, 56, 56, 0}, {"max"}, dev_bufs[0], dev_bufs[1]);
}

TEST(Cudnn, external_function_cudnn3) {
  Context::Global().ResetNameId();

  common::CudaModuleTester tester;

  auto* A_host        = common::BufferBuilder(Float(32), {2, 1000}).set_random().Build();
  auto* B_host        = common::BufferBuilder(Float(32), {2, 1000}).set_random().Build();
  auto* C_target_host = common::BufferBuilder(Float(32), {2, 1000}).set_zero().Build();

  auto* A_dev = tester.CreateDeviceBuffer(A_host);
  auto* B_dev = tester.CreateDeviceBuffer(B_host);

  cinn_buffer_t* dev_bufs[2];
  for (int i = 0; i < 2; i++) dev_bufs[i] = new cinn_buffer_t;
  dev_bufs[0]->memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->memory = reinterpret_cast<uint8_t*>(B_dev);

  runtime::cuda::cinn_gpu_cudnn_softmax({2, 1000, -1}, dev_bufs[0], dev_bufs[1]);
}
#endif
}  // namespace backends
}  // namespace cinn

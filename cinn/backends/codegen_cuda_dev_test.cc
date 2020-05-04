#include "cinn/backends/codegen_cuda_dev.h"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <vector>

#include "cinn/backends/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/runtime/cuda/cuda_module.h"

namespace cinn {
namespace backends {

TEST(CodeGenCUDA, basic) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->WithBuffer();

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", {A, B, C});

  auto compiled = codegen.Compile(func);

  std::cout << compiled << std::endl;
}

TEST(CodeGenCUDA, Module_output) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->WithBuffer();

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", {A, B, C});

  Module module("module", target);
  module.Append(func);

  Outputs outputs;
  outputs = outputs.cuda_source("generated1.cu");
  codegen.Compile(module, outputs);
}

TEST(CodeGenCUDA, compile_run_jit) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->WithBuffer();

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", {A, B, C});

  Module module("module", target);
  module.Append(func);

  Outputs outputs;
  outputs          = outputs.cuda_source("generated1.cu");
  auto source_code = codegen.Compile(module);

  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  // compile the code
  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  // launch the kernel

  CUdeviceptr Ad, Bd, Cd;
  cuMemAlloc(&Ad, M.as_int32() * N.as_int32() * sizeof(float));
  cuMemAlloc(&Bd, M.as_int32() * N.as_int32() * sizeof(float));
  cuMemAlloc(&Cd, M.as_int32() * N.as_int32() * sizeof(float));

  int num_elements = M.as_int32() * N.as_int32();

  std::vector<float> host_data1(num_elements, 0);
  std::vector<float> host_data2(num_elements, 0);
  std::vector<float> host_data3(num_elements, 0);
  for (float& v : host_data1) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT
  for (float& v : host_data2) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT

  CUDA_CALL(
      cudaMemcpy(reinterpret_cast<void*>(Ad), host_data1.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(reinterpret_cast<void*>(Bd), host_data2.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(M.as_int32(), 1, 1);
  dim3 block(N.as_int32(), 1, 1);
  cuda_module.LaunchKernel(0, "elementwise_add_kernel", grid, block, args);

  CUDA_CALL(
      cudaMemcpy(host_data3.data(), reinterpret_cast<void*>(Cd), num_elements * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < M.as_int32(); i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      int offset = i * N.as_int32() + j;
      EXPECT_NEAR(host_data3[offset], host_data1[offset] * host_data2[offset], 1e-5);
    }
  }
}

TEST(CodeGenCUDA, jit_dynamic_shape) {
  Var M("M");
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->WithBuffer();

  auto [M_outer, M_inner] = C->stage()->Split(0, 32);  // M/32, 32
  C->stage()->Reorder({
      M_inner,
      C->stage()->axis(2),
      M_outer,
  });

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  auto func = Lower("elementwise_add", {A, B, C}, {M});
  LOG(INFO) << "func:\n" << func;

  CodeGenCUDA_Dev codegen(target);

  Module module("module", target);
  module.Append(func);

  Outputs outputs;
  outputs          = outputs.cuda_source("generated1.cu");
  auto source_code = codegen.Compile(module);

  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  // compile the code
  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  // launch the kernel

  const int m            = 200;
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
    host_data1[i] = rand() / INT_MAX;  // NOLINT
    host_data2[i] = rand() / INT_MAX;  // NOLINT
  }

  CUDA_CALL(
      cudaMemcpy(reinterpret_cast<void*>(Ad), host_data1.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(reinterpret_cast<void*>(Bd), host_data2.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

  void* args[] = {const_cast<int*>(&m), &Ad, &Bd, &Cd};

  dim3 grid(32, 1, 1);
  dim3 block(N.as_int32(), 1, 1);
  cuda_module.LaunchKernel(0, "elementwise_add_kernel", grid, block, args);

  CUDA_CALL(
      cudaMemcpy(host_data3.data(), reinterpret_cast<void*>(Cd), num_elements * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      int offset = i * N.as_int32() + j;
      EXPECT_NEAR(host_data3[offset], host_data1[offset] * host_data2[offset], 1e-5);
    }
  }
}

}  // namespace backends
}  // namespace cinn

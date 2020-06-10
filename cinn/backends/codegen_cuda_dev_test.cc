#include "cinn/backends/codegen_cuda_dev.h"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <tuple>
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

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  Outputs outputs;
  outputs = outputs.cuda_source("generated1.cu");
  codegen.Compile(builder.Build(), outputs);
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

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  Outputs outputs;
  outputs          = outputs.cuda_source("generated1.cu");
  auto source_code = codegen.Compile(builder.Build());

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

class ElementwiseTester {
 public:
  Expr N{212};
  Var M{"M"};

  explicit ElementwiseTester(const std::string& fn_name) : fn_name_(fn_name) {}

  std::tuple<Placeholder<float>, Placeholder<float>, ir::Tensor> BuildNet() {
    Target target;

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
    C->WithBuffer();

    return std::make_tuple(A, B, C);
  }

  void Test(Placeholder<float>& A,  // NOLINT
            Placeholder<float>& B,  // NOLINT
            ir::Tensor& C,          // NOLINT
            std::vector<int> grid_sizes,
            std::vector<int> block_sizes) {
    Var M("M");
    auto func = Lower(fn_name_, {A, B, C}, {M});
    LOG(INFO) << "func:\n" << func;

    Target target;
    Module::Builder builder("module", target);
    builder.AddFunction(func);

    CodeGenCUDA_Dev codegen(target);
    Outputs outputs;
    outputs          = outputs.cuda_source("generated1.cu");
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

    cuda_module.LaunchKernel(0, fn_name_ + "_kernel", grid, block, args);

    CUDA_CALL(cudaMemcpy(
        host_data3.data(), reinterpret_cast<void*>(Cd), num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < N.as_int32(); j++) {
        int offset = i * N.as_int32() + j;
        if (i == 0 && j < 2) {
          LOG(INFO) << host_data3[offset];
        }
        EXPECT_NEAR(host_data3[offset], host_data1[offset] * host_data2[offset], 1e-5);
      }
    }
  }

 private:
  std::string fn_name_;
};

TEST(CodeGenCUDA, jit_dynamic_shape0) {
  ElementwiseTester tester("elementwise_base");
  auto [A, B, C] = tester.BuildNet();  // NOLINT

  auto [M_outer, M_inner] = C->stage()->Split(0, 32);  // M/32, 32 NOLINT
  C->stage()->Reorder({
      M_inner,
      C->stage()->axis(2),
      M_outer,
  });

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  tester.Test(A, B, C, {32}, {tester.N.as_int32()});
}

TEST(CodeGenCUDA, jit_dynamic_shape1) {
  ElementwiseTester tester("elementwise1");
  auto [A, B, C] = tester.BuildNet();  // NOLINT

  auto [M_outer, M_inner] = C->stage()->Split(0, 32);  // M/32, 32 NOLINT
  auto [N_outer, N_inner] = C->stage()->Split(2, 32);  // M/32, 32 NOLINT
  C->stage()->Reorder({
      M_inner,
      N_inner,
      M_outer,
      N_outer,
  });

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  tester.Test(A, B, C, {32}, {32});
}

TEST(CodeGenCUDA, jit_dynamic_shape2) {
  ElementwiseTester tester("elementwise2");
  auto [A, B, C] = tester.BuildNet();  // NOLINT

  auto [M_outer, M_inner] = C->stage()->Split(0, 32);  // M/32, 32 NOLINT
  auto [N_outer, N_inner] = C->stage()->Split(2, 3);   // M/32, 32 NOLINT
  C->stage()->Reorder({
      M_inner,
      N_inner,
      M_outer,
      N_outer,
  });

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  tester.Test(A, B, C, {32}, {3});
}

}  // namespace backends
}  // namespace cinn

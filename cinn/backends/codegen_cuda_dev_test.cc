#include "cinn/backends/codegen_cuda_dev.h"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <tuple>
#include <vector>

#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/use_extern_funcs.h"

namespace cinn {
namespace backends {

std::tuple<CUdeviceptr, CUdeviceptr, CUdeviceptr, std::vector<float>, std::vector<float>, std::vector<float>>
CreateNVMemory(int M, int N) {
  CUDA_CALL(cudaThreadSynchronize());

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

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  // compile the code
  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  auto [Ad, Bd, Cd, host_data1, host_data2, host_data3] = CreateNVMemory(M.as_int32(), N.as_int32());

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(M.as_int32(), 1, 1);
  dim3 block(N.as_int32(), 1, 1);
  cuda_module.LaunchKernel(0, "elementwise_add_kernel", grid, block, args);

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

    CUDA_CALL(cudaFree(reinterpret_cast<void*>(Ad)))
    CUDA_CALL(cudaFree(reinterpret_cast<void*>(Bd)))
    CUDA_CALL(cudaFree(reinterpret_cast<void*>(Cd)))
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

TEST(CodeGenCUDA, host) {
  auto [Ad, Bd, Cd, host_data1, host_data2, host_data3] = CreateNVMemory(100, 200);

  ElementwiseTester tester("elementwise_host_test");
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

  Var M("M");
  auto func = Lower("fn", {A, B, C}, {M});

  LOG(INFO) << "func:\n" << func;

  Target target;
  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto module = builder.Build();
  Expr expr(module);

  auto [host_module, device_module] = SplitCudaAndHostModule(module);  // NOLINT
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
  fn_kernel = cuda_module.GetFunction(0, "fn_kernel");
  CHECK(fn_kernel);

  LOG(INFO) << "fn_kernel: " << fn_kernel;

  RuntimeSymbolRegistry::Global().Register("fn_kernel_ptr_", reinterpret_cast<void*>(&fn_kernel));
  RuntimeSymbolRegistry::Global().Register("fn_kernel_stream_ptr_", reinterpret_cast<void*>(&stream));

  // compile host
  {
    auto jit = SimpleJIT::Create();
    jit->Link<CodeGenCUDA_Host>(host_module, false);

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

    A_buf->host_memory = reinterpret_cast<uint8_t*>(Ad);
    B_buf->host_memory = reinterpret_cast<uint8_t*>(Bd);
    C_buf->host_memory = reinterpret_cast<uint8_t*>(Cd);

    CUDA_CALL(cudaThreadSynchronize());

    // call the kernel
    auto comp = reinterpret_cast<void (*)(cinn_pod_value_t*, int)>(fn_ptr);
    cinn_pod_value_t args[10];
    cinn_pod_value_t M_arg(M.as_int32());
    cinn_pod_value_t A_arg(A_buf);
    cinn_pod_value_t B_arg(B_buf);
    cinn_pod_value_t C_arg(C_buf);
    cinn_args_construct(args, 4, &M_arg, &A_arg, &B_arg, &C_arg);

    comp(args, 4);

    CUDA_CALL(cudaThreadSynchronize());

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

}  // namespace backends
}  // namespace cinn

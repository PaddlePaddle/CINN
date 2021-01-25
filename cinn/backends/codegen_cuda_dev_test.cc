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
#include "cinn/ir/ir_printer.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"
#include "cinn/runtime/use_extern_funcs.h"

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

  Target target;

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

TEST(CodeGenCUDA2, compile_run_jit2) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("X", {M, N});
  Placeholder<float> B("Y", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});
  std::vector<ir::Tensor> readers{C};
  auto B_cache = stages[B]->CacheRead2("local", readers, stages);
  stages[B_cache]->Split(0, 10);
  stages[C]->Split(0, 10);
  stages[B_cache]->Bind(0, "blockIdx.x");
  stages[B_cache]->Bind(1, "threadIdx.x");
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[B_cache]->SyncThreads({C}, stages);
  stages[B_cache]->ComputeAt2(stages[C], 0);
  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add3", stages, {A, B, C});

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled CacheRead2 sync code:\n\n\n" << source_code;

  std::string source_target = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void elementwise_add3(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ C)
{
  float _Y_read_cache [ ((1 * (((1 * 100) * 200) / 10)) / 10) ];
  float* Y_read_cache = _Y_read_cache;
  if ((blockIdx.x < 10)) {
  {
    if ((threadIdx.x < 10)) {
    {
      for (int32_t j = 0; j < 200; j += 1) {
        Y_read_cache[j] = Y[((2000 * blockIdx.x) + ((200 * threadIdx.x) + j))];
      };
    }
    };
    __syncthreads();
    if ((threadIdx.x < 10)) {
    {
      for (int32_t j = 0; j < 200; j += 1) {
        C[((2000 * blockIdx.x) + ((200 * threadIdx.x) + j))] = (X[((2000 * blockIdx.x) + ((200 * threadIdx.x) + j))] * Y_read_cache[j]);
      };
    }
    };
  }
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

  auto [Ad, Bd, Cd, host_data1, host_data2, host_data3] = CreateNVMemory(M.as_int32(), N.as_int32());

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(10, 1, 1);
  dim3 block(10, 1, 1);
  cuda_module.LaunchKernel(0, "elementwise_add3", grid, block, args);

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

TEST(CodeGenCUDA, compile_run_jit) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});
  std::vector<ir::Tensor> readers{C};
  auto B_cache = stages[B]->CacheRead2("local", readers, stages);
  stages[B_cache]->Bind(0, "blockIdx.x");
  stages[B_cache]->Bind(1, "threadIdx.x");
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[B_cache]->SyncThreads({C}, stages);
  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_add", stages, {A, B, C});

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled CacheRead2 code:\n\n\n" << source_code;

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
  float _B_read_cache [ ((1 * (((1 * 100) * 200) / 100)) / 200) ];
  float* B_read_cache = _B_read_cache;
  if ((blockIdx.x < 100)) {
  {
    if ((threadIdx.x < 200)) {
    {
      B_read_cache[0] = B[((200 * blockIdx.x) + threadIdx.x)];
    }
    };
  }
  };
  __syncthreads();
  if ((blockIdx.x < 100)) {
  {
    if ((threadIdx.x < 200)) {
    {
      C[((200 * blockIdx.x) + threadIdx.x)] = (A[((200 * blockIdx.x) + threadIdx.x)] * B_read_cache[0]);
    }
    };
  }
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

  auto [Ad, Bd, Cd, host_data1, host_data2, host_data3] = CreateNVMemory(M.as_int32(), N.as_int32());

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

TEST(CodeGenCUDA3, compile_run_jit3) {
  Expr M(32);
  Expr N(32);
  Expr K(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A1", {M, K});
  Placeholder<float> B("B1", {N, K});

  auto k1 = Var(K.as_int32(), "k1");
  auto CC = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(j, k1), {k1}); }, "C1");

  auto stages = CreateStages({CC});
  std::vector<ir::Tensor> readers{CC};

  auto C = stages[CC]->CacheWrite2("local", stages);

  stages[C]->Split(1, 2);
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  stages[CC]->Split(1, 2);
  stages[CC]->Bind(0, "blockIdx.x");
  stages[CC]->Bind(1, "threadIdx.x");

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("mul_cache_write", stages, {A, B, C}, {}, {}, nullptr, target);

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "compiled CacheWrite+InitReduce code:\n\n\n" << source_code;

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
  float _C1_cache_write_out [ ((1 * (((1 * 32) * 32) / 32)) / 16) ];
  float* C1_cache_write_out = _C1_cache_write_out;
  float* C1_cache_write_out__reduce_init = _C1_cache_write_out;
  if ((blockIdx.x < 32)) {
  {
    if ((threadIdx.x < 16)) {
    {
      for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
        C1_cache_write_out__reduce_init[j_inner] = 0;
        for (int32_t k1 = 0; k1 < 32; k1 += 1) {
          C1_cache_write_out[j_inner] = (C1_cache_write_out[j_inner] + (A1[((32 * blockIdx.x) + k1)] * B1[((32 * j_inner) + ((64 * threadIdx.x) + k1))]));
        };
      };
    }
    };
  }
  };
  if ((blockIdx.x < 32)) {
  {
    if ((threadIdx.x < 16)) {
    {
      for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
        C1[((32 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))] = C1_cache_write_out[j_inner];
      };
    }
    };
  }
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

  auto [Ad, Bd, Cd, host_data1, host_data2, host_data3] = CreateNVMemory(M.as_int32(), N.as_int32());

  // launch the kernel

  void* args[] = {&Ad, &Bd, &Cd};

  dim3 grid(32, 1, 1);
  dim3 block(16, 1, 1);
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
    Target target;

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
  auto [A, B, C] = tester.BuildNet();  // NOLINT

  auto stages = CreateStages({C});

  auto [M_outer, M_inner] = stages[C]->Split(0, 32);  // M/32, 32 NOLINT
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
  auto [A, B, C] = tester.BuildNet();  // NOLINT

  auto stages = CreateStages({C});

  auto [M_outer, M_inner] = stages[C]->Split(0, 32);  // M/32, 32 NOLINT
  auto [N_outer, N_inner] = stages[C]->Split(2, 32);  // M/32, 32 NOLINT
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

  auto [A, B, C] = tester.BuildNet();  // NOLINT

  auto stages = CreateStages({C});

  auto [M_outer, M_inner] = stages[C]->Split(0, 32);  // M/32, 32 NOLINT
  auto [N_outer, N_inner] = stages[C]->Split(2, 3);   // M/32, 32 NOLINT
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
  auto [Ad, Bd, Cd, host_data1, host_data2, host_data3] = CreateNVMemory(100, 200);

  ElementwiseTester tester("elementwise_host_test");
  auto [A, B, C] = tester.BuildNet();  // NOLINT
  auto stages    = CreateStages({C});

  auto [M_outer, M_inner] = stages[C]->Split(0, 32);  // M/32, 32 NOLINT
  auto [N_outer, N_inner] = stages[C]->Split(2, 3);   // M/32, 32 NOLINT
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
  std::vector<ir::Tensor> temp;
  auto B_cache = stages[B]->CacheRead2("shared", temp, stages);

  auto fn = Lower("fn", stages, {A, W, B, B_cache});

  LOG(INFO) << "fn:\n" << fn;
}

TEST(elementwise_add1, share_local_cache) {
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
  auto AA = stages[A]->CacheRead2("local", temp, stages);
  auto AL = stages[AA]->CacheRead2("local", temp, stages);
  // NOTE here, the CC replace the C as the output the function.
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[AA]->Bind(0, "blockIdx.x");
  stages[AA]->Bind(1, "threadIdx.x");
  stages[AL]->ComputeAt2(stages[C], 1);

  Module::Builder builder("gpu_module", common::DefaultNVGPUTarget());

  auto fn = Lower("elementwise_add1", stages, {A, B, C});

  builder.AddFunction(fn);
  auto module = builder.Build();

  // compile with device code
  CodeGenCUDA_Dev codegen(common::DefaultNVGPUTarget());
  auto source_code = codegen.Compile(builder.Build());

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
  tester("elementwise_add1", args.data(), args.size());
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
  Expr M(100);
  Expr N(20);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  auto stages = CreateStages({C});

  auto CC = stages[C]->CacheWrite2("local", stages);
  std::vector<ir::Tensor> temp{C};
  auto AA = stages[A]->CacheRead2("shared", temp, stages);
  // NOTE here, the CC replace the C as the output the function.

  stages[CC]->Bind(0, "blockIdx.x");
  stages[CC]->Bind(1, "threadIdx.x");

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  stages[AA]->Bind(0, "blockIdx.x");
  stages[AA]->Bind(1, "threadIdx.x");

  Module::Builder builder("gpu_module", common::DefaultNVGPUTarget());

  auto fn = Lower("elementwise_add0", stages, {A, B, CC}, {}, {AA, C});

  ASSERT_EQ(fn->temp_bufs.size(), 2UL);
  builder.AddFunction(fn);
  auto module = builder.Build();

  auto [host_module, device_module] = SplitCudaAndHostModule(module);  // NOLINT
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
  auto BL = stages[B]->CacheWrite2("local", stages);
  auto AA = stages[Apad]->CacheRead2("shared", temp, stages);
  auto WW = stages[W]->CacheRead2("shared", temp, stages);
  auto AL = stages[AA]->CacheRead2("local", temp, stages);
  auto WL = stages[WW]->CacheRead2("local", temp, stages);

  stages[Apad]->ComputeInline();

  // tile consts
  const int tile         = 8;
  const int num_thread   = 8;
  const int block_factor = tile * num_thread;
  const int step         = 8;
  const int vthread      = 2;

  auto hi = stages[B]->axis(0);
  auto wi = stages[B]->axis(1);
  auto fi = stages[B]->axis(2);
  auto ni = stages[B]->axis(3);
  auto bz = stages[B]->Fuse(hi, wi);

  poly::Iterator by, bx, ty, tx;
  std::tie(by, fi) = stages[B]->Split(fi, block_factor);  // NOLINT
  std::tie(bx, ni) = stages[B]->Split(ni, block_factor);  // NOLINT

  poly::Iterator tyz, txz;
  std::tie(tyz, fi) = stages[B]->Split(fi, vthread);
  std::tie(txz, ni) = stages[B]->Split(ni, vthread);
  std::tie(ty, fi)  = stages[B]->Split(fi, num_thread);
  std::tie(tx, ni)  = stages[B]->Split(ni, num_thread);
  stages[B]->Reorder({bz, by, bx, tyz, txz, ty, tx, fi, ni});

  LOG(INFO) << Lower("conv", stages, {A, W, BL}, {}, {AA, WW, AL, WL, B});
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

  auto AL = stages[A]->CacheRead2("local", temp, stages);
  stages[C]->Split(1, 10);
  stages[AL]->Split(1, 10);

  stages[AL]->ComputeAt2(stages[C], 1);
  stages[C]->Bind(0, "threadIdx.x");
  stages[C]->Bind(1, "blockIdx.x");

  Target target;
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
  float _A_read_cache [ ((1 * (((1 * 100) * 200) / 100)) / 20) ];
  float* A_read_cache = _A_read_cache;
  if ((threadIdx.x < 100)) {
  {
    if ((blockIdx.x < 20)) {
    {
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        A_read_cache[j_inner] = A[((10 * blockIdx.x) + ((200 * threadIdx.x) + j_inner))];
      };
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        C[((10 * blockIdx.x) + ((200 * threadIdx.x) + j_inner))] = (A_read_cache[j_inner] + B[((10 * blockIdx.x) + ((200 * threadIdx.x) + j_inner))]);
      };
    }
    };
  }
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
  Expr M(100);
  Expr N(200);

  auto create_module = [&] {
    Context::Global().ResetNameId();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M - 2, N}, [&](Expr i, Expr j) { return A(i, j) + A(i + 1, j) + A(i + 2, j) + B(i, j); }, "C");

    auto stages = CreateStages({C});
    auto AL     = stages[A]->CacheRead("local", {C}, stages);

    stages[C]->Split(1, 10);
    stages[AL]->Split(1, 10);
    stages[AL]->ComputeAt(stages[C], 1, poly::Stage::ComputeAtKind::kComputeAtAuto, A->name);

    return std::make_tuple(A, B, C, AL, stages);
  };
  {
    auto [A, B, C, AL, stages] = create_module();  // NOLINT
    auto fn                    = Lower("fn1", stages, {A, B, C}, {}, {AL});
    CodeGenC codegen_c(common::DefaultHostTarget());
    codegen_c.SetInlineBuiltinCodes(false);

    Module::Builder builder("module", common::DefaultHostTarget());
    builder.AddFunction(fn);

    auto c_source_code = codegen_c.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
    std::cout << "C source code:\n" << c_source_code << std::endl;
  }

  auto [A, B, C, AL, stages] = create_module();  // NOLINT
  stages[C]->Bind(0, "threadIdx.x");
  stages[C]->Bind(1, "blockIdx.x");

  Target target;
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn1", stages, {A, B, C}, {}, {AL});
  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  std::cout << "CUDA source:\n" << source_code << std::endl;

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
  float _A_read_cache [ 3 * 10 ];
  float* A_read_cache = _A_read_cache;
  if ((threadIdx.x < 98)) {
  {
    if ((blockIdx.x < 20)) {
    {
      if (((((threadIdx.x >= 0) && (threadIdx.x <= 97)) && (blockIdx.x >= 0)) && (blockIdx.x <= 19))) {
        for (int32_t i = 0; i < 3; i += 1) {
          for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
            A_read_cache[((10 * i) + j_inner)] = A[((10 * blockIdx.x) + ((200 * i) + ((200 * threadIdx.x) + j_inner)))];
          };
        };
      };
      for (int32_t i = 0; i < 10; i += 1) {
        C[((10 * blockIdx.x) + ((200 * threadIdx.x) + i))] = (A_read_cache[i] + (A_read_cache[(10 + i)] + (A_read_cache[(20 + i)] + B[((10 * blockIdx.x) + ((200 * threadIdx.x) + i))])));
      };
    }
    };
  }
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
                  1e-5);
    }
  }

  cuMemFree(reinterpret_cast<CUdeviceptr>(A_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(B_dev));
  cuMemFree(reinterpret_cast<CUdeviceptr>(C_dev));
}

TEST(GetTransformedLevel, basic) {
  Expr M(10), N(10);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute({M, N}, [&](Expr i, Expr j) { return A(i, j); });
  auto D = Compute({M, N}, [&](Expr i, Expr j) { return C(i, j); });

  auto stages = CreateStages({C, D});

  // No ComputeAt, the GetTransformedLevel just returns the level without change.
  ASSERT_EQ(stages[C]->GetTransformedLevel(0), 0);

  stages[C]->ComputeAt2(stages[D], 1);
  ASSERT_EQ(stages[C]->GetTransformedLevel(0), 0 + 1 + 1);
}

// JIT test precision for the basic elementwise add
void TestElementwiseAddPrecisionBasic(
    const ir::Module& module,
    const std::string& fn_name,
    Expr M,
    Expr N,
    std::function<float(float, float)> elem_cal = [](float a, float b) { return a; }) {
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
  for (int i = 0; i < M.as_int32() - 2; i++) {
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
    Context::Global().ResetNameId();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [&](Expr i, Expr j) { return A(i, j); }, "C");
    auto stages = CreateStages({A, B, C});
    std::vector<ir::Tensor> temp{C};
    auto AL = stages[A]->CacheRead2("shared", temp, stages);

    stages[C]->Split(1, 10);
    stages[AL]->Split(1, 10);
    stages[C]->Bind(0, "blockIdx.x");
    stages[C]->Bind(1, "threadIdx.x");
    stages[AL]->ComputeAt2(stages[C], 1);

    return std::make_tuple(A, B, C, AL, stages);
  };

  auto [A, B, C, AL, stages] = create_module();  // NOLINT
  Target target;
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
  __shared__ float _A_read_cache [ (((1 * 100) * 200) / 100) ];
  float* A_read_cache = _A_read_cache;
  if ((blockIdx.x < 100)) {
  {
    if ((threadIdx.x < 20)) {
    {
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        A_read_cache[((10 * threadIdx.x) + j_inner)] = A[((200 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))];
      };
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        C[((200 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))] = A_read_cache[((10 * threadIdx.x) + j_inner)];
      };
    }
    };
  }
  };
}

}
)ROC";

  LOG(INFO) << "GPU thread config: " << fn->cuda_axis_info;

  ASSERT_EQ(utils::Trim(target_source), source_code);

  TestElementwiseAddPrecisionBasic(builder.Build(), "fn2", M, N);
}

// This test is meaningless for a cache read, we just check that the syncthreads is automatically inserted even without
// ComputeAt.
TEST(ElementwiseAdd, cache_read_shared_no_compute_at) {
  // Make a small shape, because the shared memory is small.
  Expr M(40);
  Expr N(40);

  auto create_module = [&] {
    Context::Global().ResetNameId();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [&](Expr i, Expr j) { return A(i, j); }, "C");

    auto stages = CreateStages({A, B, C});
    std::vector<ir::Tensor> temp{C};
    auto AL = stages[A]->CacheRead2("shared", temp, stages);

    stages[C]->Split(1, 10);
    stages[AL]->Split(1, 10);

    stages[C]->Bind(0, "blockIdx.x");
    stages[C]->Bind(1, "threadIdx.x");
    stages[AL]->Bind(0, "blockIdx.x");
    stages[AL]->Bind(1, "threadIdx.x");

    return std::make_tuple(A, B, C, AL, stages);
  };

  auto [A, B, C, AL, stages] = create_module();  // NOLINT
  Target target;
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn3", stages, {A, B, C}, {}, {AL});

  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  std::cout << "CUDA source3:\n" << source_code << std::endl;

  auto target_source = R"ROC(
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn3(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  __shared__ float _A_read_cache [ (((1 * 40) * 40) / 40) ];
  float* A_read_cache = _A_read_cache;
  if ((blockIdx.x < 40)) {
  {
    if ((threadIdx.x < 4)) {
    {
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        A_read_cache[((10 * threadIdx.x) + j_inner)] = A[((40 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))];
      };
    }
    };
  }
  };
  if ((blockIdx.x < 40)) {
  {
    if ((threadIdx.x < 4)) {
    {
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        C[((40 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))] = A_read_cache[((10 * threadIdx.x) + j_inner)];
      };
    }
    };
  }
  };
}

}
)ROC";

  LOG(INFO) << "GPU thread config: " << fn->cuda_axis_info;

  ASSERT_EQ(utils::Trim(target_source), source_code);

  TestElementwiseAddPrecisionBasic(builder.Build(), "fn3", M, N);
}

TEST(ElementwiseAdd, cache_write_local) {
  // Make a small shape, because the shared memory is small.
  Expr M(40);
  Expr N(40);

  auto create_module = [&] {
    Context::Global().ResetNameId();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M, N}, [&](Expr i, Expr j) { return A(i, j); }, "C");

    auto stages = CreateStages({A, B, C});

    auto Co = stages[C]->CacheWrite2("local", stages);

    // Cache write local, the local memory can just share in a single thread, so it must ComputeAt(inside) the innermost
    // thread.
    stages[C]->ComputeAt2(stages[Co], 1);
    stages[Co]->Bind(0, "blockIdx.x");
    stages[Co]->Bind(1, "threadIdx.x");

    return std::make_tuple(A, B, C, Co, stages);
  };

  auto [A, B, C, Co, stages] = create_module();  // NOLINT

  CodeGenCUDA_Dev codegen(common::DefaultNVGPUTarget());

  auto fn = Lower("fn4", stages, {A, B, Co}, {}, {C});

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
void fn4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  float _C_cache_write_out [ ((1 * (((1 * 40) * 40) / 40)) / 40) ];
  float* C_cache_write_out = _C_cache_write_out;
  if ((blockIdx.x < 40)) {
  {
    if ((threadIdx.x < 40)) {
    {
      C_cache_write_out[0] = A[((40 * blockIdx.x) + threadIdx.x)];
      C[((40 * blockIdx.x) + threadIdx.x)] = C_cache_write_out[0];
    }
    };
  }
  };
}

}
)ROC";

  LOG(INFO) << "GPU thread config: " << fn->cuda_axis_info;

  ASSERT_EQ(utils::Trim(target_source), source_code);

  TestElementwiseAddPrecisionBasic(builder.Build(), "fn4", M, N);
}

TEST(Cuda, external_function) {
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

  auto [A, B, C, stages] = create_module();  // NOLINT
  Target target;
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn5", stages, {A, B, C});

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
void fn5(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  if ((blockIdx.x < 40)) {
  {
    if ((threadIdx.x < 4)) {
    {
      for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
        C[((40 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))] = (cinn_nvgpu_tanh_fp32(A[((40 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))]) + cinn_nvgpu_cos_fp32(B[((40 * blockIdx.x) + ((10 * threadIdx.x) + j_inner))]));
      };
    }
    };
  }
  };
}

}
)ROC";

  LOG(INFO) << "GPU thread config: " << fn->cuda_axis_info;

  ASSERT_EQ(utils::Trim(target_source), source_code);

  TestElementwiseAddPrecisionBasic(
      builder.Build(), "fn5", M, N, [](float a, float b) { return std::tanh(a) + std::cos(b); });
}

}  // namespace backends
}  // namespace cinn

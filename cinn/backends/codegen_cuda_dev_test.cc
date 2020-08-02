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
#include "cinn/common/cuda_test_helper.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/test_helper.h"
#include "cinn/ir/ir_printer.h"
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
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  C->stage()->Bind(0, "blockIdx.x");
  C->stage()->Bind(1, "threadIdx.x");

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

  C->stage()->Bind(0, "blockIdx.x");
  C->stage()->Bind(1, "threadIdx.x");

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

  C->stage()->Bind(0, "blockIdx.x");
  C->stage()->Bind(1, "threadIdx.x");

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

  auto [M_outer, M_inner] = C->stage()->Split(0, 32);  // M/32, 32 NOLINT
  C->stage()->Reorder({
      M_inner,
      C->stage()->axis(2),
      M_outer,
  });

  C->stage()->Bind(0, "blockIdx.x");
  C->stage()->Bind(1, "threadIdx.x");

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

  C->stage()->Bind(0, "blockIdx.x");
  C->stage()->Bind(1, "threadIdx.x");

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

  C->stage()->Bind(0, "blockIdx.x");
  C->stage()->Bind(1, "threadIdx.x");

  tester.Test(A, B, C, {32}, {3});
}

TEST(CodeGenCUDA, jit_host_call_cuda_kernel) {
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

  C->stage()->Bind(0, "blockIdx.x");
  C->stage()->Bind(1, "threadIdx.x");

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
  IS->WithBuffer();
  auto FS = Compute(
      ir::Tensor(filter)->shape,
      [=](Expr c0, Expr c1, Expr w, Expr h) -> Expr { return filter(c0, c1, w, h); },
      "cache_filter");
  FS->WithBuffer();

  auto output = Compute({Expr(batch), Expr(in_channel), Expr(out_height), Expr(out_width)},
                        [=](Var b, Var c, Var i, Var j) -> Expr {
                          auto expr = IS(b, c, i * stride + di, j * stride + dj) * FS(c, c, di, dj);
                          return Sum(expr);
                        },
                        "output",
                        {di, dj});
  output->WithBuffer();

  output->stage()->Fuse(0, 1);
  output->stage()->Fuse(0, 1);
  output->stage()->Split(0, 20);

  // similar to
  // s[Output].bind(Output.op.axis[0], block_y)
  // s[Output].bind(Output.op.axis[1], block_x)
  // output->stage()->GpuThreads(output->stage()->ith_iterator(1), output->stage()->ith_iterator(0));
  // IS->stage()->GpuThreads(IS->stage()->ith_iterator(1), IS->stage()->ith_iterator(0));
  // FS->stage()->GpuThreads(FS->stage()->ith_iterator(1), FS->stage()->ith_iterator(0));
  // IS->stage()->ComputeAt(output->stage(), 1, poly::Stage::kComputeAtBefore);

  // const int block_size = 32;

  // // schedule
  // // bx1, _ = s[Output].split(Output.op.axis[2], factor=blocking_h)
  // auto [bx1, _bx1] = output->stage()->Split(2, block_size);  // NOLINT
  // // x2, _ = s[Output].split(Output.op.axis[3], factor=blocking_w)
  // auto [bx2, _bx2] = output->stage()->Split(3, block_size);

  // // assign one 32x32 block to one cuda block.
  // auto by = output->stage()->Fuse(0, 1);
  // auto bx = output->stage()->Fuse(bx1, bx2);
  // output->stage()->GpuBlocks(bx, by);

  auto fn = Lower("fn", {input, filter, IS, FS, output});

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

  auto B = Compute({out_size, out_size, out_channel, batch},
                   [=](Expr yy, Expr xx, Expr ff, Expr nn) -> Expr {
                     return Sum(Apad(yy * stride + ry, xx * stride + rx, rc, nn) * W(ry, rx, rc, ff));
                   },
                   "B",
                   {rc, ry, rx});
  B->WithBuffer();

  B->stage()->CacheRead("shared", {B});

  auto fn = Lower("fn", {A, W, B});

  LOG(INFO) << "fn:\n" << fn;
}

// Test the basic elementwise_add kernel with share cache set.
// A JIT is created to test JIT call GPU.
TEST(elementwise_add, share_local_cache) {
  // TODO(Superjomn) fix this, make cache read work
  return;
  Expr M(100);
  Expr N(20);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  auto AA = A->stage()->CacheRead("shared", {C});
  // NOTE here, the CC replace the C as the output the function.
  auto CC = C->stage()->CacheWrite("local");

  C->stage()->Bind(0, "blockIdx.x");
  C->stage()->Bind(1, "threadIdx.x");

  A->stage()->Bind(0, "blockIdx.x");
  AA->stage()->Bind(1, "threadIdx.x");

  CC->stage()->Bind(0, "blockIdx.x");
  CC->stage()->Bind(1, "threadIdx.x");

  Target target;
  Module::Builder builder("gpu_module", target);

  auto fn = Lower("elementwise_add", {A, B, CC}, {}, {AA, C}, &builder);

  ASSERT_EQ(fn->temp_bufs.size(), 2UL);

  auto module = builder.Build();

  auto [host_module, device_module] = SplitCudaAndHostModule(module);  // NOLINT
  for (auto& func : host_module.functions()) {
    LOG(INFO) << "host:\n" << func;
  }

  for (auto& func : device_module.functions()) {
    LOG(INFO) << "device:\n" << func;
  }

  // compile with device code
  CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());

  LOG(INFO) << "device source code:\n" << source_code;

  using runtime::cuda::CUDAModule;

  backends::NVRTC_Compiler compiler;

  auto ptx = compiler(source_code);
  LOG(INFO) << "PTX:\n" << ptx;
  CHECK(!ptx.empty());

  CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);
  auto fn_kernel = cuda_module.GetFunction(0, "elementwise_add_kernel");
  CHECK(fn_kernel);

  // Register to JIT
  void* stream = nullptr;
  RuntimeSymbolRegistry::Global().Register("elementwise_add_kernel_ptr_", reinterpret_cast<void*>(&fn_kernel));
  RuntimeSymbolRegistry::Global().Register("elementwise_add_kernel_stream_ptr_", reinterpret_cast<void*>(&stream));

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

  CUDA_CALL(
      cudaMemcpy(reinterpret_cast<void*>(Ad), host_data1.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(reinterpret_cast<void*>(Bd), host_data2.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

  auto test_precision = [&] {
    CUDA_CALL(cudaMemcpy(
        host_data3.data(), reinterpret_cast<void*>(Cd), num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < N.as_int32(); j++) {
        int offset = i * N.as_int32() + j;
        if (i == 0 && j < 2) {
          LOG(INFO) << host_data3[offset];
        }
        ASSERT_NEAR(host_data3[offset], host_data1[offset] + host_data2[offset], 1e-5);
      }
    }
  };

  {  // test by call the compiled kernel directly
    void* args[] = {&Ad, &Bd, &Cd};

    dim3 grid(M.as_int32(), 1, 1);
    dim3 block(N.as_int32(), 1, 1);

    cuda_module.LaunchKernel(0, "elementwise_add_kernel", grid, block, args);

    test_precision();
  }

  {  // test by trigger the host jit
    auto jit = SimpleJIT::Create();
    jit->Link<CodeGenCUDA_Host>(host_module, false);

    auto fn_ptr = jit->Lookup("elementwise_add");
    CHECK(fn_ptr);

    cinn_buffer_t* A_buf =
        cinn_buffer_new(cinn_x86_device, cinn_float32_t(), std::vector<int>{{M.as_int32(), N.as_int32()}});
    cinn_buffer_t* B_buf =
        cinn_buffer_new(cinn_x86_device, cinn_float32_t(), std::vector<int>{{M.as_int32(), N.as_int32()}});
    cinn_buffer_t* C_buf =
        cinn_buffer_new(cinn_x86_device, cinn_float32_t(), std::vector<int>{{M.as_int32(), N.as_int32()}});

    A_buf->host_memory = reinterpret_cast<uint8_t*>(Ad);
    B_buf->host_memory = reinterpret_cast<uint8_t*>(Bd);
    C_buf->host_memory = reinterpret_cast<uint8_t*>(Cd);

    CUDA_CALL(cudaDeviceSynchronize());

    // call the kernel
    auto comp = reinterpret_cast<void (*)(cinn_pod_value_t*, int)>(fn_ptr);

    auto args = common::ArgsBuilder().Add(A_buf).Add(B_buf).Add(C_buf).Build();

    comp(args.data(), args.size());

    CUDA_CALL(cudaDeviceSynchronize());
  }

  CUDA_CALL(cudaFree(reinterpret_cast<void*>(Ad)))
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(Bd)))
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(Cd)))
}

TEST(Conv, basic_add_cache) {
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
      {in_size + 2 * pad, in_size + 2 * pad, in_channel, batch}, [=](Expr yy, Expr xx, Expr cc, Expr nn) -> Expr {
        return common::select(common::and_all({yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size}),
                              A(yy - pad, xx - pad, cc, nn),
                              common::make_const(Float(32), 0));
      });

  auto AA = Compute(
      Apad->shape, [=](const std::vector<Expr>& dims) -> Expr { return Apad(dims); }, "AA");
  auto WW = Compute(
      W->shape, [=](const std::vector<Expr>& dims) { return W(dims); }, "WW");
  AA->WithBuffer("shared");
  WW->WithBuffer("shared");

  auto AL = Compute(
      AA->shape, [=](const std::vector<Expr>& dims) -> Expr { return AA(dims); }, "AL");
  auto WL = Compute(
      WW->shape, [=](const std::vector<Expr>& dims) -> Expr { return WW(dims); }, "WL");

  AL->WithBuffer("local");
  WL->WithBuffer("local");

  Var rc(in_channel, "rc");
  Var ry(kernel, "ry");
  Var rx(kernel, "rx");

  auto BL = Compute({out_size, out_size, out_channel, batch},
                    [=](Expr yy, Expr xx, Expr ff, Expr nn) -> Expr {
                      return Sum(AA(yy * stride + ry, xx * stride + rx, rc, nn) * WW(ry, rx, rc, ff));
                    },
                    "BL",
                    {rc, ry, rx});
  BL->WithBuffer("local");

  auto B = Compute(
      BL->shape, [=](const std::vector<Expr>& dims) -> Expr { return BL(dims); }, "B");
  B->WithBuffer();

  int tile         = 8;
  int num_thread   = 8;
  int block_factor = tile * num_thread;
  int step         = 8;
  int vthread      = 2;

  auto hi = BL->stage()->ith_dim_name(0);
  auto wi = BL->stage()->ith_dim_name(1);
  auto fi = BL->stage()->ith_dim_name(2);
  auto ni = BL->stage()->ith_dim_name(3);

  auto bz        = BL->stage()->Fuse(hi, wi);
  auto [by, fi1] = BL->stage()->Split(fi, block_factor);
  auto [bx, ni1] = BL->stage()->Split(ni, block_factor);

  BL->stage()->Bind(2, "blockIdx.x");
  BL->stage()->Bind(1, "blockIdx.y");
  BL->stage()->Bind(0, "blockIdx.z");

  auto fn = Lower("fn", {A, W, AA, WW, AL, WL, BL, B});

  LOG(INFO) << "fn:\n" << fn;
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

  auto B = Compute({out_size, out_size, out_channel, batch},
                   [=](Expr yy, Expr xx, Expr ff, Expr nn) {
                     return Sum(Apad(yy * stride + ry, xx * stride + rx, rc, nn) * W(ry, rx, rc, ff));
                   },
                   "B",
                   {rc, ry, rx} /*reduce axis*/);

  // auto fn = Lower("conv", {A, W, B});
  // LOG(INFO) << fn;

  // blocking

  Apad->stage()->ComputeInline();

  auto AA = Apad->stage()->CacheRead("shared", {B});
  auto WW = W->stage()->CacheRead("shared", {B});
  auto AL = AA->stage()->CacheRead("local", {B});
  auto WL = WW->stage()->CacheRead("local", {B});
  auto BL = B->stage()->CacheWrite("local");

  // tile consts
  const int tile         = 8;
  const int num_thread   = 8;
  const int block_factor = tile * num_thread;
  const int step         = 8;
  const int vthread      = 2;

  auto hi = B->stage()->axis(0);
  auto wi = B->stage()->axis(1);
  auto fi = B->stage()->axis(2);
  auto ni = B->stage()->axis(3);
  auto bz = B->stage()->Fuse(hi, wi);

  poly::Iterator by, bx, ty, tx;
  std::tie(by, fi) = B->stage()->Split(fi, block_factor);  // NOLINT
  std::tie(bx, ni) = B->stage()->Split(ni, block_factor);  // NOLINT

  poly::Iterator tyz, txz;
  std::tie(tyz, fi) = B->stage()->Split(fi, vthread);
  std::tie(txz, ni) = B->stage()->Split(ni, vthread);
  std::tie(ty, fi)  = B->stage()->Split(fi, num_thread);
  std::tie(tx, ni)  = B->stage()->Split(ni, num_thread);
  B->stage()->Reorder({bz, by, bx, tyz, txz, ty, tx, fi, ni});

  LOG(INFO) << Lower("conv", {A, W, BL}, {}, {AA, WW, AL, WL, B});
}

TEST(ElementwiseAdd, cache_read_local) {
  Context::Global().ResetNameId();

  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");
  C->stage()->Split(1, 10);

  auto AL = A->stage()->CacheRead("local", {C});
  AL->stage()->Split(1, 10);

  AL->stage()->ComputeAt(C->stage(), 1, poly::Stage::ComputeAtKind::kComputeAtAuto, A->name);
  C->stage()->Bind(0, "threadIdx.x");
  C->stage()->Bind(1, "blockIdx.x");

  Target target;
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn0", {A, B, C}, {}, {AL});

  Module::Builder builder("module", target);
  builder.AddFunction(fn);

  auto module      = builder.Build();
  auto source_code = codegen.Compile(module);
  LOG(INFO) << "source:\n" << source_code;

  std::string source_target = R"ROC(
extern "C" {

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn0_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  float _A_read_cache_3 [ 1 * 10 ];
  float* A_read_cache_3 = _A_read_cache_3;
  if ((threadIdx.x < 100)) {
  {
    if ((blockIdx.x < 20)) {
    {
      if (((((threadIdx.x >= 0) && (threadIdx.x <= 99)) && (blockIdx.x >= 0)) && (blockIdx.x <= 19))) {
        for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
          A_read_cache_3[j_inner] = A[((10 * blockIdx.x) + ((200 * threadIdx.x) + j_inner))];
        };
      };
      for (int32_t i = 0; i < 10; i += 1) {
        C[((10 * blockIdx.x) + ((200 * threadIdx.x) + i))] = (A_read_cache_3[i] + B[((10 * blockIdx.x) + ((200 * threadIdx.x) + i))]);
      };
    }
    };
  }
  };
}

}
)ROC";
  ASSERT_EQ(utils::Trim(source_target), source_code);

  auto [host_module, device_module] = SplitCudaAndHostModule(module);  // NOLINT

  backends::NVRTC_Compiler compiler;

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
  dev_bufs[0]->host_memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->host_memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->host_memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args                = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("fn0", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->host_memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->host_memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->host_memory);
  auto* B_mem        = reinterpret_cast<float*>(B_host->host_memory);
  for (int i = 0; i < C_target_host->num_elements(); i++) {
    ASSERT_NEAR(C_target_mem[i], A_mem[i] + B_mem[i], 1e-5);
  }
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
    C->stage()->Split(1, 10);

    auto AL = A->stage()->CacheRead("local", {C});
    AL->stage()->Split(1, 10);

    AL->stage()->ComputeAt(C->stage(), 1, poly::Stage::ComputeAtKind::kComputeAtAuto, A->name);

    return std::make_tuple(A, B, C, AL);
  };
  {
    auto [A, B, C, AL] = create_module();  // NOLINT
    auto fn            = Lower("fn1", {A, B, C}, {}, {AL});
    CodeGenC codegen_c(common::DefaultHostTarget());
    codegen_c.SetInlineBuiltinCodes(false);

    Module::Builder builder("module", common::DefaultHostTarget());
    builder.AddFunction(fn);

    auto c_source_code = codegen_c.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
    std::cout << "C source code:\n" << c_source_code << std::endl;
  }

  auto [A, B, C, AL] = create_module();  // NOLINT
  C->stage()->Bind(0, "threadIdx.x");
  C->stage()->Bind(1, "blockIdx.x");

  Target target;
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn1", {A, B, C}, {}, {AL});
  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  std::cout << "CUDA source:\n" << source_code << std::endl;

  std::string source_target = R"ROC(
extern "C" {

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn1_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  float _A_read_cache_3 [ 3 * 10 ];
  float* A_read_cache_3 = _A_read_cache_3;
  if ((threadIdx.x < 98)) {
  {
    if ((blockIdx.x < 20)) {
    {
      if (((((threadIdx.x >= 0) && (threadIdx.x <= 97)) && (blockIdx.x >= 0)) && (blockIdx.x <= 19))) {
        for (int32_t i = 0; i < 3; i += 1) {
          for (int32_t j_inner = 0; j_inner < 10; j_inner += 1) {
            A_read_cache_3[((10 * i) + j_inner)] = A[((10 * blockIdx.x) + ((200 * i) + ((200 * threadIdx.x) + j_inner)))];
          };
        };
      };
      for (int32_t i = 0; i < 10; i += 1) {
        C[((10 * blockIdx.x) + ((200 * threadIdx.x) + i))] = (A_read_cache_3[i] + (A_read_cache_3[(10 + i)] + (A_read_cache_3[(20 + i)] + B[((10 * blockIdx.x) + ((200 * threadIdx.x) + i))])));
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
  dev_bufs[0]->host_memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->host_memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->host_memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args                = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("fn1", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->host_memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->host_memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->host_memory);
  auto* B_mem        = reinterpret_cast<float*>(B_host->host_memory);
  for (int i = 0; i < M.as_int32() - 2; i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      ASSERT_NEAR(C_target_mem[i * N.as_int32() + j],
                  A_mem[i * N.as_int32() + j] + A_mem[(i + 1) * N.as_int32() + j] + A_mem[(i + 2) * N.as_int32() + j] +
                      B_mem[i * N.as_int32() + j],
                  1e-5);
    }
  }
}

TEST(ElementwiseAdd, cache_read2) {
  Expr M(100);
  Expr N(200);

  auto create_module = [&] {
    Context::Global().ResetNameId();

    Placeholder<float> A("A", {M, N});
    Placeholder<float> B("B", {M, N});

    auto C = Compute(
        {M - 2, N}, [&](Expr i, Expr j) { return A(i, j) + A(i + 1, j) + A(i + 2, j) + B(i, j); }, "C");
    C->stage()->Split(1, 10);

    auto AL = A->stage()->CacheRead("local", {C});
    AL->stage()->Split(1, 10);

    AL->stage()->ComputeAt(C->stage(), 0, poly::Stage::ComputeAtKind::kComputeAtAuto, A->name);

    return std::make_tuple(A, B, C, AL);
  };
  {
    auto [A, B, C, AL] = create_module();  // NOLINT
    auto fn            = Lower("fn1", {A, B, C}, {}, {AL});
    CodeGenC codegen_c(common::DefaultHostTarget());
    codegen_c.SetInlineBuiltinCodes(false);

    Module::Builder builder("module", common::DefaultHostTarget());
    builder.AddFunction(fn);

    auto c_source_code = codegen_c.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
    std::cout << "C source code:\n" << c_source_code << std::endl;
  }

  auto [A, B, C, AL] = create_module();  // NOLINT
  C->stage()->Bind(0, "threadIdx.x");
  C->stage()->Bind(1, "blockIdx.x");

  Target target;
  CodeGenCUDA_Dev codegen(target);

  auto fn = Lower("fn1", {A, B, C}, {}, {AL});
  Module::Builder builder("module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  auto source_code = codegen.Compile(builder.Build());
  std::cout << "CUDA source:\n" << source_code << std::endl;

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
  dev_bufs[0]->host_memory = reinterpret_cast<uint8_t*>(A_dev);
  dev_bufs[1]->host_memory = reinterpret_cast<uint8_t*>(B_dev);
  dev_bufs[2]->host_memory = reinterpret_cast<uint8_t*>(C_dev);
  auto args                = common::ArgsBuilder().Add(dev_bufs[0]).Add(dev_bufs[1]).Add(dev_bufs[2]).Build();

  CUDA_CALL(cudaDeviceSynchronize());
  tester("fn1", args.data(), args.size());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(C_target_host->host_memory),
                       C_dev,
                       C_target_host->num_elements() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  auto* C_target_mem = reinterpret_cast<float*>(C_target_host->host_memory);
  auto* A_mem        = reinterpret_cast<float*>(A_host->host_memory);
  auto* B_mem        = reinterpret_cast<float*>(B_host->host_memory);
  for (int i = 0; i < M.as_int32() - 2; i++) {
    for (int j = 0; j < N.as_int32(); j++) {
      ASSERT_NEAR(C_target_mem[i * N.as_int32() + j],
                  A_mem[i * N.as_int32() + j] + A_mem[(i + 1) * N.as_int32() + j] + A_mem[(i + 2) * N.as_int32() + j] +
                      B_mem[i * N.as_int32() + j],
                  1e-5);
    }
  }
}

}  // namespace backends
}  // namespace cinn

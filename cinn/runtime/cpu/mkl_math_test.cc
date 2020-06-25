#include <gtest/gtest.h>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/cinn.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/test_helper.h"
#include "cinn/runtime/cpu/host_intrinsics.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"

namespace cinn {
namespace runtime {
namespace cpu {

cinn_buffer_t *CreateBuffer(const std::vector<int> shape, bool random = true) {
  if (random) {
    return common::BufferBuilder(Float(32), shape).set_random().Build();
  }
  return common::BufferBuilder(Float(32), shape).set_zero().Build();
}

void TestCallElementwise(const std::string &fn_name, float (*fn_runtime)(float), bool is_elementwise) {
  Expr M(10);
  Expr N(10);
  Placeholder<float> x("x", {M, N});

  ir::Tensor out;

  std::vector<ir::Tensor> lower_args({x});
  if (is_elementwise) {
    out = Compute(
        {M, N}, [&](Var i, Var j) -> Expr { return lang::CallExtern(fn_name, {x(i, j)}); }, fn_name + "_out");
    lower_args.push_back(out);
  } else {
    auto comp_out = Compute(
        {Expr(1)}, [&]() -> Expr { return lang::CallExtern(fn_name, {x}); }, fn_name + "_out");
    out = comp_out->TupleGet(0);
    out->WithBuffer(Float(32));
    lower_args.push_back(out);
    lower_args.push_back(comp_out);
  }

  auto target = common::DefaultHostTarget();
  target.arch = Target::Arch::X86;
  lang::Module::Builder builder("module0", target);
  auto func = Lower("fn", lower_args);
  builder.AddFunction(func);

  LOG(INFO) << "func:\n" << func;

  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("fn");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  auto *A_buf = CreateBuffer({10, 10});
  auto *B_buf = CreateBuffer({10, 10}, false);

  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg};
  fn_(args, 2);

  auto *ad = reinterpret_cast<float *>(A_buf->host_memory);
  auto *bd = reinterpret_cast<float *>(B_buf->host_memory);
  for (int i = 0; i < A_buf->num_elements(); i++) {
    ASSERT_NEAR(bd[i], fn_runtime(ad[i]), 1e-5);
  }
}

TEST(mkl_math, tanh_fp32) { TestCallElementwise("cinn_cpu_tanh_fp32", cinn_cpu_tanh_fp32, true); }
TEST(mkl_math, ceil_fp32) { TestCallElementwise("cinn_cpu_ceil_fp32", cinn_cpu_ceil_fp32, true); }
TEST(mkl_math, floor_fp32) { TestCallElementwise("cinn_cpu_floor_fp32", cinn_cpu_floor_fp32, true); }
TEST(mkl_math, exp_fp32) { TestCallElementwise("cinn_cpu_exp_fp32", cinn_cpu_exp_fp32, true); }
TEST(mkl_math, tanh_v_fp32) { TestCallElementwise("cinn_mkl_tanh_v_fp32", cinn_cpu_tanh_fp32, false); }

TEST(cinn_cpu_mkl_gemm_fp32, test) {
  Expr M(30);
  Expr N(20);
  Expr K(40);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  auto call = Compute(
      {Expr(1)},
      [=]() -> Expr {
        return lang::CallExtern("cinn_cpu_mkl_gemm_fp32",
                                {
                                    common::make_one<float>(),   // alpha
                                    M,                           // M
                                    N,                           // N
                                    K,                           // K
                                    common::make_bool(false),    // ta
                                    common::make_bool(false),    // tb
                                    M,                           // lda
                                    K,                           // ldb
                                    M,                           // ldc
                                    common::make_zero<float>(),  // beta
                                    A.tensor(),                  // A
                                    B.tensor(),                  // B
                                });
      },
      "extern_call");

  auto out = call->TupleGet(0);
  out->WithBuffer(Float(32));

  auto target = common::DefaultHostTarget();
  target.arch = Target::Arch::X86;
  lang::Module::Builder builder("module0", target);
  auto func = Lower("fn", {A, B, out, call});
  builder.AddFunction(func);

  LOG(INFO) << "func:\n" << func;

  auto jit    = backends::SimpleJIT::Create();
  auto module = builder.Build();

  jit->Link(module, /*optimize=*/true);
  auto fn     = jit->Lookup("fn");
  auto fn_ptr = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  // test with real data
  auto *A_buf = common::BufferBuilder(Float(32), {M.as_int32(), K.as_int32()}).set_random().Build();
  auto *B_buf = common::BufferBuilder(Float(32), {K.as_int32(), N.as_int32()}).set_random().Build();
  auto *C_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_zero().Build();

  auto args = common::ArgsBuilder().Add(A_buf).Add(B_buf).Add(C_buf).Build();

  fn_ptr(args.data(), args.size());

  cinn_buffer_free(nullptr, A_buf);
  cinn_buffer_free(nullptr, B_buf);
  cinn_buffer_free(nullptr, C_buf);
}

}  // namespace cpu
}  // namespace runtime
}  // namespace cinn

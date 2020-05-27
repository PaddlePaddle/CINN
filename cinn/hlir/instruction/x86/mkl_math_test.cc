#include <gtest/gtest.h>

#include "cinn/backends/llvm/simple_orc_jit.h"
#include "cinn/cinn.h"
#include "cinn/hlir/instruction/x86/mkl_math_registors.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace x86 {

cinn_buffer_t *CreateBuffer(const std::vector<int> shape, bool random = true) {
  auto *buf = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), shape, 32);
  cinn_buffer_malloc(nullptr, buf);
  float *Ad = reinterpret_cast<float *>(buf->host_memory);

  for (int i = 0; i < buf->num_elements(); i++) {
    Ad[i] = random ? static_cast<float>(rand()) / RAND_MAX : 0.f;  // NOLINT
  }
  return buf;
}

TEST(mkl_math, tanh_v_fp32) {
  RegisterMklMath();

  Expr M(10);
  Expr N(10);
  Placeholder<float> x("x", {M, N});

  auto tanh_out = Compute(
      {Expr(1)}, [&]() -> Expr { return lang::CallExtern("cinn_mkl_tanh_v_fp32", {x}); }, "tanh_out");
  auto res = tanh_out->TupleGet(0);
  res->WithBuffer(Float(32));

  auto target = common::DefaultHostTarget();
  target.arch = Target::Arch::X86;
  lang::Module::Builder builder("module0", target);
  auto func = Lower("fn", {x, tanh_out, res});
  builder.AddFunction(func);

  auto jit = backends::SimpleOrcJit::Create();

  jit->Link(builder.Build(), /*optimize=*/false);
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
    ASSERT_NEAR(bd[i], __cinn_host_tanh_fp32(ad[i]), 1e-5);
  }
}

}  // namespace x86
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

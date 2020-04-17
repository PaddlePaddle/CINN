#include "cinn/backends/llvm/simple_orc_jit.h"

#include <gtest/gtest.h>
#include <llvm/IR/Function.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <tuple>

#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/module.h"
#include "cinn/lang/placeholder.h"
#include "cinn/optim/optimize.h"

namespace cinn {
namespace backends {

namespace {
auto CreateTestBuffer() {
  auto *A = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {100, 32}, 32);
  auto *B = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {100, 32}, 32);
  auto *C = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {100, 32}, 32);
  cinn_buffer_malloc(nullptr, A);
  cinn_buffer_malloc(nullptr, B);
  cinn_buffer_malloc(nullptr, C);
  float *Ad = reinterpret_cast<float *>(A->host_memory);
  float *Bd = reinterpret_cast<float *>(B->host_memory);

  for (int i = 0; i < A->num_elements(); i++) {
    Ad[i] = i;
    Bd[i] = i;
  }

  float *Cd = reinterpret_cast<float *>(C->host_memory);
  CHECK_EQ(C->num_elements(), A->num_elements());

  return std::make_tuple(A, B, C);
}

auto CreateTestCinnModule() {
  ir::Expr M(100);
  ir::Expr N(32);
  lang::Placeholder<float> A("A", {M, N});
  lang::Placeholder<float> B("B", {M, N});

  lang::Buffer C_buf(Float(32));
  auto C = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);

  common::Target target;
  target.arch = common::Target::Arch ::X86;
  target.bits = common::Target::Bit ::k32;
  target.os   = common::Target::OS ::Linux;
  lang::Module module("module1", target);

  auto funcs = lang::Lower("elementwise_add", {A, B, C});

  // auto func = optim::Optimize(funcs);

  module.Append(ir::LoweredFunc(funcs.As<ir::_LoweredFunc_>()));
  return module;
}
}  // namespace

TEST(llvm_test01, elementwise_add) {
  auto jit = backends::SimpleOrcJit::Create();

  auto [a, b, c] = CreateTestBuffer();  // NOLINT

  auto module = CreateTestCinnModule();

  jit->set_ir_file(Context::Global().runtime_llvm_ir_file());
  jit->Link(module, /*optimize=*/true);

  auto elementwise_add_addr = jit->Lookup("elementwise_add");
  auto elementwise_add      = reinterpret_cast<void (*)(void *, int32_t)>(elementwise_add_addr);
  cinn_pod_value_t a_arg(a), b_arg(b), c_arg(c);
  cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};
  elementwise_add(args, 3);

  float *ad = reinterpret_cast<float *>(a->host_memory);
  float *bd = reinterpret_cast<float *>(b->host_memory);
  float *cd = reinterpret_cast<float *>(c->host_memory);

  for (int i = 0; i < c->num_elements(); i++) {
    EXPECT_EQ(ad[i] + bd[i], cd[i]);
  }
}

}  // namespace backends
}  // namespace cinn

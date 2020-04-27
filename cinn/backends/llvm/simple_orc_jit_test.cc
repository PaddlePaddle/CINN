#include "cinn/backends/llvm/simple_orc_jit.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Function.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "cinn/backends/llvm/cinn_runtime_llvm_ir.h"
#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/module.h"
#include "cinn/lang/placeholder.h"
#include "cinn/optim/optimize.h"

namespace cinn {
namespace backends {

const int kM = 100;
const int kN = 32;

namespace {
auto CreateTestBuffer() {
  auto *A = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
  auto *B = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
  auto *C = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
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
  ir::Expr M(kM);
  ir::Expr N(kN);
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

TEST(llvm, module_call_lowered_func) {
  lang::Module module("some_module", common::DefaultHostTarget());
  ir::Expr M(kM);
  ir::Expr N(kN);
  {  // define fn
    lang::Placeholder<float> a("a", {M, N});
    lang::Placeholder<float> b("b", {M, N});
    auto c = lang::Compute(
        {M, N}, [&](auto i, auto j) { return a(i, j) + b(i, j); }, "c");
    c->WithBuffer();

    auto fn = lang::Lower("elementwise_add", {a, b, c}, {});
    module.Append(fn);
  }

  {  // call fn
    lang::Placeholder<float> a("a", {M, N});
    lang::Placeholder<float> b("b", {M, N});

    std::vector<lang::ReturnType> ret_types({lang::ReturnType{Float(32), {M, N}, "c_out"}});

    auto call_outs = lang::Call("elementwise_add", {a, b}, ret_types);
    auto c         = call_outs[0];

    // here we must call the output, so that it cal output something.

    auto main_fn = lang::Lower("main", {a, b, c}, {});
    module.Append(main_fn);
  }

  auto [ab, bb, cb] = CreateTestBuffer();  // NOLINT
  {                                        // call the function
    auto jit = backends::SimpleOrcJit::Create();

    jit->Link(module, /*optimize=*/true);
    auto elementwise_add_addr = jit->Lookup("elementwise_add");
    auto elementwise_add      = reinterpret_cast<void (*)(void *, int32_t)>(elementwise_add_addr);

    cinn_pod_value_t a_arg(ab), b_arg(bb), c_arg(cb);
    cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};

    elementwise_add(args, 3);

    for (int i = 0; i < kM; i++) {
      for (int j = 0; j < kN; j++) {
        auto *data = reinterpret_cast<float *>(cb->host_memory);
        ASSERT_NEAR(data[i * kN + j], 2 * (i * kN + j), 1e-5);
      }
    }
  }
}

}  // namespace backends
}  // namespace cinn

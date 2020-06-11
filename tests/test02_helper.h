
#pragma once

#include <gtest/gtest.h>

#include <string>
#include <string_view>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"

namespace cinn {
namespace tests {

auto CreateMatmulBasicModule(std::string_view name, int m, int n, int k) -> lang::Module {
  Expr M(m);
  Expr N(n);
  Expr K(k);
  cinn::Placeholder<float> A("A", {M, K});
  cinn::Placeholder<float> B("B", {K, N});

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();

  cinn::Buffer C_buf(Float(32));
  cinn::Var k1(K.as_int32(), "k1");
  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k1) * B(k1, j)); }, "C", {k1});
  C->Bind(C_init->buffer);

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  std::string module_name(name);
  Module::Builder builder(module_name, target);

  auto func = Lower("matmul_basic", {A, B, C, C_init});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateExecutionEngine(const cinn::lang::Module &module) {
  auto engine = cinn::backends::ExecutionEngine::Create({});
  engine->Link(module);
  return engine;
}

auto CreateSimpleJit(const cinn::lang::Module &module) {
  auto jit = cinn::backends::SimpleJIT::Create();
  jit->Link(module, true);

  return jit;
}
}  // namespace tests
}  // namespace cinn

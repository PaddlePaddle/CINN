
#pragma once

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"

namespace cinn {
namespace tests {

auto CreateMatmulBasicModule(Target target, int m, int n, int k) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k));

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C_buf = Buffer(Float(32));
  auto k1    = Var(K.as_int32(), "k1");

  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k1) * B(k1, j)); }, "C", {k1});
  C->Bind(C_init->buffer);

  Module::Builder builder("module_basic", target);

  auto func = Lower("matmul_basic", {A, B, C, C_init});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulTileModule(Target target, int m, int n, int k) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k));

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C_buf = Buffer(Float(32));
  auto k1    = Var(K.as_int32(), "k1");

  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k1) * B(k1, j)); }, "C", {k1});
  C->Bind(C_init->buffer);

  C->stage()->Tile(0, 1, 4, 4);

  Module::Builder builder("module_tile", target);

  auto func = Lower("matmul_tile", {A, B, C, C_init});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulSplitModule(Target target, int m, int n, int k) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k));

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C_buf = Buffer(Float(32));
  auto k1    = Var(K.as_int32(), "k1");

  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k1) * B(k1, j)); }, "C", {k1});
  C->Bind(C_init->buffer);

  auto c_poly_iterators = [&C](auto &&... args) {
    std::vector<poly::Iterator> iters;
    (iters.push_back(C->stage()->ith_iterator(args)), ...);
    return iters;
  };
  C->stage()->Split(2, 16);
  C->stage()->Reorder(c_poly_iterators(1, 0, 2, 3));

  Module::Builder builder("module_split", target);

  auto func = Lower("matmul_split", {A, B, C, C_init});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulBlockModule(Target target, int m, int n, int k) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k));

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();
  auto C_buf = Buffer(Float(32));
  auto k1    = Var(K.as_int32(), "k1");

  auto C = Compute({M, N}, [&](Var i, Var j) { return Sum(A(i, k1) * B(k1, j)); }, "C", {k1});
  C->Bind(C_init->buffer);
  constexpr int bn                          = 32;
  auto [i_outer, i_inner, j_outer, j_inner] = C->stage()->Tile(0, 1, bn, bn);
  auto [k_outer, k_inner]                   = C->stage()->Split(k1->name, 4);
  C->stage()->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});

  Module::Builder builder("module_block", target);

  auto func = Lower("matmul_block", {A, B, C, C_init});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateCinnMatmulModule(const std::string &name, Target target, int m, int n, int k) {
  if (name == "basic") {
    return CreateMatmulBasicModule(target, m, n, k);
  } else if (name == "tile") {
    return CreateMatmulTileModule(target, m, n, k);
  } else if (name == "split") {
    return CreateMatmulSplitModule(target, m, n, k);
  } else if (name == "block") {
    return CreateMatmulBlockModule(target, m, n, k);
  }
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

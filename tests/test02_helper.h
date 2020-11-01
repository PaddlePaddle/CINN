
#pragma once

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

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

  auto k1 = Var(K.as_int32(), "k1");
  auto C  = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(k1, j), {k1}); }, "C");

  auto stages = CreateStages({C});

  Module::Builder builder("module_basic", target);

  auto func = Lower("matmul_basic", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulTileModule(Target target, int m, int n, int k) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k));

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto k1 = Var(K.as_int32(), "k1");
  auto C  = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(k1, j), {k1}); }, "C");

  auto stages = CreateStages({C});

  stages[C]->Tile(0, 1, 4, 4);

  Module::Builder builder("module_tile", target);

  auto func = Lower("matmul_tile", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulSplitModule(Target target, int m, int n, int k) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k));

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto k1 = Var(K.as_int32(), "k1");
  auto C  = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(k1, j), {k1}); }, "C");

  auto stages = CreateStages({C});

  auto c_poly_iterators = [&](auto &&... args) {
    std::vector<poly::Iterator> iters;
    (iters.push_back(stages[C]->ith_iterator(args)), ...);
    return iters;
  };
  stages[C]->Split(2, 16);
  stages[C]->Reorder(c_poly_iterators(1, 0, 2, 3));

  Module::Builder builder("module_split", target);

  auto func = Lower("matmul_split", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulBlockModule(Target target, int m, int n, int k) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k));

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto k1 = Var(K.as_int32(), "k1");
  auto C  = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(k1, j), {k1}); }, "C");

  auto stages = CreateStages({C});

  constexpr int bn                          = 32;
  auto [i_outer, i_inner, j_outer, j_inner] = stages[C]->Tile(0, 1, bn, bn);  // NOLINT
  auto [k_outer, k_inner]                   = stages[C]->Split(k1->name, 4);  // NOLINT
  stages[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});

  Module::Builder builder("module_block", target);

  auto func = Lower("matmul_block", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulVectorizeModule(Target target, int m, int n, int k) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k));

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  Var k0(K.as_int32(), "k0");

  int bn = 32;

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k0) * B(k0, j), {k0}); }, "C");

  auto stages = CreateStages({C});

  {
    auto [i_outer, i_inner, j_outer, j_inner] = stages[C]->Tile(0, 1, bn, bn);
    auto [k_outer, k_inner]                   = stages[C]->Split("k0", 4);
    stages[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
    stages[C]->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_vectorize", target);
  auto func = Lower("matmul_vectorize", stages, {A, B, C});

  builder.AddFunction(func);

  return builder.Build();
}

ir::Module CreateMatmulLoopPermutation(Target target, int m, int n, int k_) {
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os   = Target::OS::Linux;

  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k_));

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  int bn = 32;

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  auto stages = CreateStages({C});

  // Blocking by loop tiling.
  {
    auto [i_outer, i_inner, j_outer, j_inner] = stages[C]->Tile(0, 1, bn, bn);  // NOLINT
    auto [k_outer, k_inner]                   = stages[C]->Split("k0", 4);      // NOLINT

    stages[C]->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});

    stages[C]->Vectorize(j_inner, 8);
    stages[C]->Unroll(5);
  }

  Module::Builder builder("module_loop_permutation", target);
  auto func = Lower("matmul_loop_permutation", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

ir::Module CreateMatmulArrayPacking(Target target, int m, int n, int k_) {
  auto [M, N, K] = std::make_tuple(Expr(m), Expr(n), Expr(k_));

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  Expr bn(32);

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  auto packedB = Compute(
      {N / bn, K, bn}, [&](Expr x, Expr y, Expr z) { return B(y, x * bn + z); }, "packedB");
  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return ReduceSum(A(i, k) * packedB(j / bn, k, j % bn), {k}); }, "C");

  auto stages = CreateStages({C_init, C});

  stages[C]->ShareBufferWith(stages[C_init]);
  stages[C]->CtrlDepend(C_init);
  stages[packedB]->Vectorize(2, 8);

  {
    auto [i_outer, i_inner, j_outer, j_inner] = stages[C]->Tile(0, 1, bn.as_int32(), bn.as_int32());  // NOLINT
    auto [k_outer, k_inner]                   = stages[C]->Split("k0", 4);                            // NOLINT

    stages[C]->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
    stages[C]->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_array_packing", target);
  auto func = Lower("matmul_array_packing", stages, {A, B, C, packedB});

  builder.AddFunction(func);

  return builder.Build();
}

// TODO(Superjomn) To refactor this, strange to use if-else here.
auto CreateCinnMatmulModule(const std::string &name, Target target, int m, int n, int k) {
  if (name == "basic") {
    return CreateMatmulBasicModule(target, m, n, k);
  } else if (name == "tile") {
    return CreateMatmulTileModule(target, m, n, k);
  } else if (name == "split") {
    return CreateMatmulSplitModule(target, m, n, k);
  } else if (name == "block") {
    return CreateMatmulBlockModule(target, m, n, k);
  } else if (name == "vectorize") {
    return CreateMatmulVectorizeModule(target, m, n, k);
  } else if (name == "loop_permutation") {
    return CreateMatmulLoopPermutation(target, m, n, k);
  } else if (name == "array_packing") {
    return CreateMatmulArrayPacking(target, m, n, k);
  }
  { CINN_NOT_IMPLEMENTED }
}

auto CreateExecutionEngine(const cinn::ir::Module &module) {
  auto engine = cinn::backends::ExecutionEngine::Create({});
  engine->Link(module);
  return engine;
}

auto CreateSimpleJit(const cinn::ir::Module &module) {
  auto jit = cinn::backends::SimpleJIT::Create();
  jit->Link(module, true);

  return jit;
}
}  // namespace tests
}  // namespace cinn

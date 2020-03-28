#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"

namespace cinn {
using poly::Iterator;

const int M = 1024;
const int N = 1024;
const int K = 1024;

TEST(test02_matmul, basic) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K, "k");
  Buffer C_buf(Float(32));

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->Bind(C_buf);
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j), k); }, "C", k);
  C->Bind(C_buf);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  {
    Module module("module1", target);
    auto funcs = Lower("matmul", {A, B, C, C_init});
    ASSERT_EQ(funcs.size(), 1UL);

    auto func = Optimize(funcs.front());
    module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
    // module.Append(C_buf);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul.h").c_source("./test02_matmul.cc");
    compiler.Compile(module, outputs);
  }

  // Tile
  {
    C->stage()->Tile(0, 1, 4, 4);

    Module module("module2", target);
    auto funcs = Lower("matmul_tile", {A, B, C, C_init});
    ASSERT_EQ(funcs.size(), 1UL);

    auto func = Optimize(funcs.front());
    module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
    // module.Append(C_buf);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul_tile.h").c_source("./test02_matmul_tile.cc");
    compiler.Compile(module, outputs);
  }
}

TEST(matmul, Split) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K, "k");
  Buffer C_buf(Float(32));

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->Bind(C_buf);
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j), k); }, "C", k);
  C->Bind(C_buf);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Iterator i0, i1;
  std::tie(i0, i1) = C->stage()->Split(2, 16);
  std::vector<Iterator> iterators({C->stage()->ith_iterator(1),
                                   C->stage()->ith_iterator(0),
                                   C->stage()->ith_iterator(2),
                                   C->stage()->ith_iterator(3)});
  C->stage()->Reorder(iterators);

  Module module("module3", target);
  auto funcs = Lower("matmul_split", {A, B, C, C_init});
  ASSERT_EQ(funcs.size(), 1UL);

  auto func = Optimize(funcs.front());
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_split.h").c_source("./test02_matmul_split.cc");
  compiler.Compile(module, outputs);
}

TEST(matmul, Blocking) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K, "k");
  Buffer C_buf(Float(32));

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->Bind(C_buf);
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j), k); }, "C", k);
  C->Bind(C_buf);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  // Blocking by loop tiling.
  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    Iterator k_outer, k_inner;
    std::tie(k_outer, k_inner) = C->stage()->Split("k", 4);

    C->stage()->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
  }

  Module module("module_block", target);
  auto funcs = Lower("matmul_block", {A, B, C, C_init});
  ASSERT_EQ(funcs.size(), 1UL);

  auto func = Optimize(funcs.front());
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_block.h").c_source("./test02_matmul_block.cc");
  compiler.Compile(module, outputs);
}

TEST(matmul, Vectorization) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K, "k");
  Buffer C_buf(Float(32));

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->Bind(C_buf);
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j), k); }, "C", k);
  C->Bind(C_buf);
  // ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  // Blocking by loop tiling.
  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    Iterator k_outer, k_inner;
    std::tie(k_outer, k_inner) = C->stage()->Split("k", 4);

    C->stage()->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});

    C->stage()->Vectorize(j_inner, 8);
  }

  Module module("module_vectorize", target);
  auto funcs = Lower("matmul_vectorize", {A, B, C, C_init});
  ASSERT_EQ(funcs.size(), 1UL);

  auto func = Optimize(funcs.front());
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_vectorize.h").c_source("./test02_matmul_vectorize.cc");
  compiler.Compile(module, outputs);
}

TEST(matmul, LoopPermutation) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K, "k");
  Buffer C_buf(Float(32));

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->Bind(C_buf);
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j), k); }, "C", k);
  C->Bind(C_buf);
  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  // Blocking by loop tiling.
  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    Iterator k_outer, k_inner;

    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    std::tie(k_outer, k_inner)                   = C->stage()->Split("k", 4);

    C_init->stage()->Vectorize(1, 8);
    C_init->stage()->Unroll(1);

    C->stage()->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});

    C->stage()->Vectorize(j_inner, 8);
    C->stage()->Unroll(5);
  }

  Module module("module_loop_permutation", target);
  auto funcs = Lower("matmul_loop_permutation", {A, B, C, C_init});
  ASSERT_EQ(funcs.size(), 1UL);

  auto func = Optimize(funcs.front());
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_loop_permutation.h").c_source("./test02_matmul_loop_permutation.cc");
  compiler.Compile(module, outputs);
}

TEST(matmul, ArrayPacking) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K, "k");
  Buffer C_buf(Float(32));

  int bn = 32;

  auto C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->Bind(C_buf);
  auto packedB = Compute(
      {N / bn, K, bn}, [&](Expr x, Expr y, Expr z) { return B(y, x * bn + z); }, "packedB");
  Buffer packedB_buf(packedB->type());
  packedB->Bind(packedB_buf);
  packedB->stage()->Vectorize(2, 8);

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return Sum(A(i, k) * packedB(j / bn, k, j % bn), k); }, "C", k);
  C->Bind(C_buf);

  ASSERT_EQ(C->buffer_depended_tensor_names().size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  {
    Iterator i_outer, i_inner, j_outer, j_inner;
    Iterator k_outer, k_inner;

    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    std::tie(k_outer, k_inner)                   = C->stage()->Split("k", 4);

    C->stage()->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
    //C->stage()->Split(j_inner, 8);
  }

  Module module("module_array_packing", target);
  auto funcs = Lower("matmul_array_packing", {A, B, C, C_init, packedB});
  ASSERT_EQ(funcs.size(), 1UL);

  auto func = Optimize(funcs.front());
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  // module.Append(funcs.front());

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_array_packing.h").c_source("./test02_matmul_array_packing.cc");
  compiler.Compile(module, outputs);
}

}  // namespace cinn

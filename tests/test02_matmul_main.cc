#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/optim/optimize.h"

namespace cinn {

const int M = 1000;
const int N = 400;
const int K = 500;

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

  poly::Iterator i0, i1;
  std::tie(i0, i1) = C->stage()->Split(2, 16);
  std::vector<poly::Iterator> iterators({C->stage()->ith_iterator(1),
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

}  // namespace cinn

#include <gtest/gtest.h>

#include "cinn/cinn.h"

namespace cinn {

TEST(test02_matmul, basic) {
  const int M = 1000;
  const int N = 400;
  const int K = 500;

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K, "k");

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return Sum(A(i, k) * B(k, j), k); }, "C", k);
  Buffer C_buf(C->type());
  C->Bind(C_buf);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module module("module1", target);

  auto funcs = Lower("matmul", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  module.Append(funcs.front());
  module.Append(C_buf);

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul.h").c_source("./test02_matmul.cc");
  compiler.Compile(module, outputs);
}

}  // namespace cinn

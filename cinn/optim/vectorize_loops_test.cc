#include "cinn/optim/vectorize_loops.h"
#include <gtest/gtest.h>
#include "cinn/cinn.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace optim {

TEST(VectorizeLoops, basic) {
  using namespace ir;
  const int M  = 100;
  const int K  = 200;
  const int N  = 500;
  const int bn = 32;
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));
  Var k(K, "k");

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return lang::Sum(A(i, k) * B(k, j), k); }, "C", k);
  C->Bind(C_buf);

  {
    poly::Iterator i_outer, i_inner, j_outer, j_inner, k_outer, k_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    std::tie(k_outer, k_inner)                   = C->stage()->Split(poly::Iterator("k"), 8);
    C->stage()->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
    C->stage()->Split(j_inner, 8);
  }

  // Code gen
  auto funcs = Lower("matmul", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  optim::VectorizeLoops(&funcs[0]->body, target);

  lang::Module module("module1", target);
  module.Append(funcs.front());
  module.Append(C_buf);

  CodeGenC codegen(target);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "out:\n" << out;
}

}  // namespace optim
}  // namespace cinn

#include "cinn/backends/codegen_c_x86.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/module.h"
#include "cinn/lang/placeholder.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/transform_polyfor_to_for.h"
#include "cinn/optim/vectorize_loops.h"

namespace cinn {
namespace backends {

TEST(CodeGenCX86, basic) {
  // create two forloops, check only one forloop is marked Vectorize.
  Context::Global().info_rgt().Clear();

  using namespace ir;  // NOLINT

  const int M  = 100;
  const int K  = 200;
  const int N  = 500;
  const int bn = 32;

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->Bind(C_buf);

  Tensor D = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "D");
  D->Bind(C_buf);

  // vectorize C, not D
  C->stage()->Vectorize(1, 16);
  C->stage()->Unroll(1);

  auto funcs = Lower("matmul", {A, B, C, D});
  CHECK_EQ(funcs.size(), 1UL);

  std::cout << "before optim\n" << funcs.front()->body << std::endl;

  funcs.front()->body = Optimize(funcs.front()->body);

  lang::Module module("module1", target);
  module.Append(funcs[0]);

  CodeGenCX86 codegen(target, CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "out:\n" << out;

  EXPECT_EQ(Context::Global().info_rgt().Get<int>("vectorized_forloop_count"), 1);
}

}  // namespace backends
}  // namespace cinn

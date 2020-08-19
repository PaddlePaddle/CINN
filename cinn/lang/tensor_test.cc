#include "cinn/lang/tensor.h"

#include <gtest/gtest.h>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/common/test_helper.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/packed_func.h"
#include "cinn/lang/placeholder.h"

namespace cinn {
namespace ir {
using utils::GetStreamCnt;
using utils::Trim;

TEST(Tensor, inlined) {
  Expr M(100), N(20);

  lang::Placeholder<float> A("A", {M, N});
  lang::Placeholder<float> B("B", {M, N});

  // C is inlined
  Tensor C = lang::Compute(
      {M, N}, [=](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->stage()->ComputeInline();

  Tensor D = lang::Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return C(i, j) * 2.f + 1.f; }, "D");

  auto funcs = lang::Lower("func_C", {A, B, D});
  std::cout << "output: \n" << funcs << std::endl;
  auto out = GetStreamCnt(funcs);
  EXPECT_EQ(Trim(out), Trim(R"ROC(
function func_C (_A, _B, _D)
{
  for (i, 100)
  {
    for (j, 20)
    {
      D[i, j] = (1 + ((2 * A[i, j]) + (2 * B[i, j])))
    }
  }
}
)ROC"));
}

TEST(Tensor, Collapse) {
  Expr M0(10), M1(20), M2(30), M3(40);

  lang::Placeholder<float> A("A", {M0, M1, M2, M3});

  // new shape: [M0*M1, M2, M3]
  auto A1 = Tensor(A).Reshape({M0 * M1, M2, M3});

  // inlined
  auto C0 = lang::Compute({M0 * M1, M2, M3}, [&](Var i, Var j, Var k) { return A1(i, j, k) * A1(i, j, k); });
  C0->stage()->ComputeInline();

  auto C = lang::Compute(
      {M0 * M1, M2, M3}, [&](Var i, Var j, Var k) { return C0(i, j, k) + 1.f; }, "C");

  auto func = lang::Lower("func", {A, C});
  LOG(INFO) << "func:\n" << func;

  lang::Module::Builder builder("module0", common::DefaultHostTarget());
  builder.AddFunction(func);

  auto jit = backends::ExecutionEngine::Create({});
  jit->Link(builder.Build());

  auto fn_addr = jit->Lookup("func");
  CHECK(fn_addr);

  auto* fn = reinterpret_cast<void (*)(void*, int32_t)>(fn_addr);

  auto* A_buf = common::BufferBuilder(Float(32), {10, 20, 30, 40}).set_random().Build();
  auto* C_buf = common::BufferBuilder(Float(32), {10, 20, 30, 40}).set_zero().Build();
  auto args   = common::ArgsBuilder().Add(A_buf).Add(C_buf).Build();

  fn(args.data(), args.size());

  // check result
  auto* A_data = reinterpret_cast<float*>(A_buf->memory);
  auto* C_data = reinterpret_cast<float*>(C_buf->memory);
  for (int i = 0; i < A_buf->num_elements(); i++) {
    ASSERT_NEAR(C_data[i], A_data[i] * A_data[i] + 1.f, 1e-5);
  }
}

TEST(Tensor, IsDependOnStatement) {
  Expr N(100);

  Placeholder<float> X("X", {N});
  auto t = Compute({N}, [&](Var i) -> Expr { return X(i); });

  ASSERT_TRUE(t->IsDependOnStatement("X"));
  ASSERT_FALSE(t->IsDependOnStatement("XXX"));
}

}  // namespace ir
}  // namespace cinn

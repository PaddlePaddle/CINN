#include "cinn/ir/tensor.h"

#include <gtest/gtest.h>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/cinn.h"
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

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C is inlined
  Tensor C = lang::Compute(
      {M, N}, [=](Var i, Var j) { return A(i, j) + B(i, j); }, "C");

  Tensor D = lang::Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return C(i, j) * 2.f + 1.f; }, "D");

  auto stages = CreateStages({D});
  stages[C]->ComputeInline();

  auto func = lang::Lower("func_C", stages, {A, B, D});
  std::cout << "output: \n" << func << std::endl;
  auto out = GetStreamCnt(func);
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

TEST(Tensor, IsDependOnStatement) {
  Expr N(100);

  Placeholder<float> X("X", {N});
  auto t = Compute({N}, [&](Var i) -> Expr { return X(i); });

  ASSERT_TRUE(t->IsDependOnStatement("X"));
  ASSERT_FALSE(t->IsDependOnStatement("XXX"));
}

TEST(Tensor, Reshape) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(100);
  Placeholder<float> A("A", {M, N});

  auto stages = CreateStages({A});

  auto A1 = A->Reshape({Expr(10), Expr(10), Expr(100)}, stages);
  auto B  = Compute(A1->shape, [=](Expr i, Expr j, Expr k) { return A1(i, j, k) * 2.f; });

  stages->InsertLazily(B);

  auto func = lang::Lower("fn", stages, {A, B});

  lang::Module::Builder builder("some_modue", common::DefaultHostTarget());
  builder.AddFunction(func);

  backends::CodeGenC codegenc(common::DefaultHostTarget());
  codegenc.SetInlineBuiltinCodes(false);
  auto source = codegenc.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  LOG(INFO) << "source:\n" << source;

  auto target_source = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void fn(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _tensor_3 = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _tensor_3);
  cinn_buffer_malloc((void*)(0), _A);
  const float* A_reshape_2 = ((const float*)(_A->memory));
  float* tensor_3 = ((float*)(_tensor_3->memory));
  for (int32_t i = 0; i < 10; i += 1) {
    for (int32_t j = 0; j < 10; j += 1) {
      for (int32_t k = 0; k < 100; k += 1) {
        tensor_3[((1000 * i) + ((100 * j) + k))] = (2 * A_reshape_2[((1000 * i) + ((100 * j) + k))]);
      };
    };
  };
  cinn_buffer_free((void*)(0), _tensor_3);
}
)ROC";

  ASSERT_EQ(Trim(target_source), Trim(source));
}

TEST(Tensor, ReshapeCopied) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(100);
  Placeholder<float> A("A", {M, N});

  auto stages = CreateStages({A});

  auto A1 = A->ReshapeCopied({Expr(10), Expr(10), Expr(100)}, stages);
  auto B  = Compute(A1->shape, [=](Expr i, Expr j, Expr k) { return A1(i, j, k) * 2.f; });

  stages->InsertLazily(B);

  lang::Module::Builder builder("some_modue", common::DefaultHostTarget());
  auto func = lang::Lower("fn", stages, {A, B}, {}, {}, &builder);

  backends::CodeGenC codegenc(common::DefaultHostTarget());
  codegenc.SetInlineBuiltinCodes(false);
  auto source = codegenc.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  LOG(INFO) << "source:\n" << source;

  auto target_source = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _A_copied_2_reshape_3 = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 10, 10, 100 }, 32/*align*/);
void fn(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _tensor_4 = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _tensor_4);
  cinn_buffer_malloc((void*)(0), _A_copied_2_reshape_3);
  const float* A_copied_2_reshape_3 = ((const float*)(_A_copied_2_reshape_3->memory));
  float* tensor_4 = ((float*)(_tensor_4->memory));
  for (int32_t i = 0; i < 10; i += 1) {
    for (int32_t j = 0; j < 10; j += 1) {
      for (int32_t k = 0; k < 100; k += 1) {
        tensor_4[((1000 * i) + ((100 * j) + k))] = (2 * A_copied_2_reshape_3[((1000 * i) + ((100 * j) + k))]);
      };
    };
  };
  cinn_buffer_free((void*)(0), _tensor_4);
}
)ROC";

  ASSERT_EQ(Trim(target_source), Trim(source));
}

}  // namespace ir
}  // namespace cinn

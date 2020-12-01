//! @h1 C++ DSL API tutorial: Matrix Multiplication
//! This tutorial will guide you through the basic usage of the C++ DSL API.

#include <gtest/gtest.h>
#include "cinn/cinn.h"

using namespace cinn;  // NOLINT
//! @IGNORE-NEXT
TEST(matmul, basic) {
  //! Declare some varialbe for latter usages.
  Expr M(100), N(200), K(50);
  Var k(K, "k0");  // the reduce axis

  //! Placeholder represents the input arguments for a computation.
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  //! Define a computation to get the result tensor C.
  auto C =
      Compute({M, N} /*domain*/, [=](Expr i, Expr j) { return ReduceSum(A(i, k) * B(k, j), {k} /*reduce axis*/); });

  //! Generate the stages to get the default schedules.
  auto stages = CreateStages({C} /*the endpoints*/);

  //! Print the generated IR, the `Lower` method just map a computation to the underlying CINN IR.
  auto fn = Lower("fn0", stages, {A, B, C} /*argument list of the fn*/);
  LOG(INFO) << "fn0:\n" << fn;
  //! This will generate the code like
  //! @ROC[c++]
  auto target_source = R"ROC(
function fn0 (_A, _B, _tensor)
{
  for (i, 100)
  {
    for (j, 200)
    {
      tensor__reduce_init[i, j] = 0
    }
  }
  for (i, 100)
  {
    for (j, 200)
    {
      for (k0, 50)
      {
        tensor[i, j] = (tensor[i, j] + (A[i, k0] * B[k0, j]))
      }
    }
  }
}
)ROC";

  //! @IGNORE-NEXT
  ASSERT_EQ(utils::GetStreamCnt(fn), utils::Trim(target_source));

  //! Print the IR as C code
  Target x;
  CodeGenC codegen(x);
  Module::Builder builder("module0", Target());
  builder.AddFunction(fn);
  codegen.SetInlineBuiltinCodes(false);  // Disable inserting the predefined runtime codes to the generated code.
  std::string C_source = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  //! @IGNORE-NEXT
  LOG(INFO) << "C:\n" << C_source;
  //! and will get some code like
  //! @ROC[c++]
  target_source = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void fn0(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _tensor = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_malloc((void*)(0), _tensor);
  const float* A = ((const float*)(_A->memory));
  const float* B = ((const float*)(_B->memory));
  float* tensor = ((float*)(_tensor->memory));
  float* tensor__reduce_init = ((float*)(_tensor->memory));
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 200; j += 1) {
      tensor__reduce_init[((200 * i) + j)] = 0;
    };
  };
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 200; j += 1) {
      for (int32_t k0 = 0; k0 < 50; k0 += 1) {
        tensor[((200 * i) + j)] = (tensor[((200 * i) + j)] + (A[((50 * i) + k0)] * B[((200 * k0) + j)]));
      };
    };
  };
  cinn_buffer_free((void*)(0), _tensor);
}
  )ROC";

  ASSERT_EQ(utils::Trim(C_source), utils::Trim(target_source));
}

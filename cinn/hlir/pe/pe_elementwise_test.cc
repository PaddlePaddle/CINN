#include <gtest/gtest.h>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/cinn.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/target.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/reduction.h"
#include "cinn/runtime/cpu/host_intrinsics.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"

namespace cinn {
namespace hlir {
namespace pe {

template <typename FuncOp, typename FuncRuntime>
void TestElementwisePE(const std::string &fn_name,
                       const FuncOp &func_op,
                       const FuncRuntime &fn_runtime,
                       Type type     = Float(32),
                       int set_value = 0) {
  Expr M(100), N(32);

  Placeholder<float> A("A", {M, N});

  auto A_out = func_op(A.tensor(), fn_name + "_out");

  auto stages = CreateStages({A.tensor(), A_out});

  Target target = common::DefaultHostTarget();
  Module::Builder builder("module0", target);
  auto func = Lower("fn", stages, {A, A_out});
  LOG(INFO) << "func:\n" << func;
  builder.AddFunction(func);

  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("fn");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf;
  if (set_value != 0) {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_val(set_value).Build();
  } else {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  }
  auto *B_buf = common::BufferBuilder(type, {M.as_int32(), N.as_int32()}).set_align(type.bits()).Build();

  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg};
  fn_(args, 2);

  auto *ad = reinterpret_cast<float *>(A_buf->memory);
  if (type.is_bool()) {
    auto *bd = reinterpret_cast<int8_t *>(B_buf->memory);
    for (int i = 0; i < A_buf->num_elements(); i++) {
      ASSERT_NEAR(bd[i], fn_runtime(ad[i]), 1e-5);
    }
  } else {
    auto *bd = reinterpret_cast<float *>(B_buf->memory);
    for (int i = 0; i < A_buf->num_elements(); i++) {
      ASSERT_NEAR(bd[i], fn_runtime(ad[i]), 1e-5);
    }
  }
}

bool isnan(float e) { return std::isnan(e); }
bool isfinite(float e) { return std::isfinite(e); }
bool isinf(float e) { return std::isinf(e); }

#define TEST_ELEMENTWISE_PE_FP32(test_name__, PE__) \
  TEST(elementwise_pe, test_name__) { TestElementwisePE("PE_Elementwise_" #test_name__ "_fp32", PE__, test_name__); }
#define TEST_ELEMENTWISE_PE_FP32_BOOL(test_name__, PE__)                                  \
  TEST(elementwise_pe, test_name__) {                                                     \
    TestElementwisePE("PE_Elementwise_" #test_name__ "_fp32", PE__, test_name__, Bool()); \
  }
#define TEST_ELEMENTWISE_PE_FP32_SET(test_name__, PE__, value__)                                      \
  TEST(elementwise_pe, test_name__) {                                                                 \
    TestElementwisePE("PE_Elementwise_" #test_name__ "_fp32", PE__, test_name__, Float(32), value__); \
  }

TEST_ELEMENTWISE_PE_FP32(exp, Exp)
TEST_ELEMENTWISE_PE_FP32(erf, Erf)
TEST_ELEMENTWISE_PE_FP32(sqrt, Sqrt)
TEST_ELEMENTWISE_PE_FP32(log, Log)
TEST_ELEMENTWISE_PE_FP32(log2, Log2)
TEST_ELEMENTWISE_PE_FP32(log10, Log10)
TEST_ELEMENTWISE_PE_FP32(floor, Floor)
TEST_ELEMENTWISE_PE_FP32(ceil, Ceil)
TEST_ELEMENTWISE_PE_FP32(round, Round)
TEST_ELEMENTWISE_PE_FP32(trunc, Trunc)
TEST_ELEMENTWISE_PE_FP32(cos, Cos)
TEST_ELEMENTWISE_PE_FP32(cosh, Cosh)
TEST_ELEMENTWISE_PE_FP32(tan, Tan)
TEST_ELEMENTWISE_PE_FP32(sin, Sin)
TEST_ELEMENTWISE_PE_FP32(sinh, Sinh)
TEST_ELEMENTWISE_PE_FP32(acos, Acos)
TEST_ELEMENTWISE_PE_FP32_SET(acosh, Acosh, 1.5)
TEST_ELEMENTWISE_PE_FP32(asin, Asin)
TEST_ELEMENTWISE_PE_FP32(asinh, Asinh)
TEST_ELEMENTWISE_PE_FP32(atan, Atan)
TEST_ELEMENTWISE_PE_FP32(atanh, Atanh)
TEST_ELEMENTWISE_PE_FP32(tanh, Tanh)
TEST_ELEMENTWISE_PE_FP32_BOOL(isnan, IsNan)
TEST_ELEMENTWISE_PE_FP32_BOOL(isfinite, IsFinite)
TEST_ELEMENTWISE_PE_FP32_BOOL(isinf, IsInf)

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

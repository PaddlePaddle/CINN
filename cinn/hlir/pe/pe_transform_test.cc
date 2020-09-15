#include <gtest/gtest.h>

#include <vector>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/cinn.h"
#include "cinn/common/target.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/pe/transform.h"
#include "cinn/runtime/cpu/host_intrinsics.h"

namespace cinn {
namespace hlir {
namespace pe {
using ir::Tensor;

TEST(MatmulPE, PE_Matmul_Test0) {
  int m = 100;
  int n = 32;
  int k = 16;
  Expr M(m), N(n), K(k);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  auto stages = CreateStages({A.tensor(), B.tensor()});
  auto C      = hlir::pe::Matmul(A.tensor(), B.tensor(), stages, false, false, 1, 1, "C");

  Target target = common::DefaultHostTarget();
  Module::Builder builder("module0", target);
  auto func = Lower("fn", stages, {A, B, C});
  builder.AddFunction(func);
  LOG(INFO) << "func:\n" << func;

  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("fn");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf = common::BufferBuilder(Float(32), {m, k}).set_random().Build();
  cinn_buffer_t *B_buf = common::BufferBuilder(Float(32), {k, n}).set_random().Build();
  cinn_buffer_t *C_buf = common::BufferBuilder(Float(32), {m, n}).set_zero().Build();

  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  auto *ad   = reinterpret_cast<float *>(A_buf->memory);
  auto *bd   = reinterpret_cast<float *>(B_buf->memory);
  auto *cd   = reinterpret_cast<float *>(C_buf->memory);
  int size_a = m;
  int size_b = n;
  int size_c = k;
  for (int i = 0; i < size_a; i++) {
    for (int j = 0; j < size_b; j++) {
      float tmp = 0;
      for (int k = 0; k < size_c; k++) {
        int index1 = i * size_c + k;
        int index2 = j + k * size_b;
        tmp += ad[index1] * bd[index2];
      }
      ASSERT_NEAR(cd[i * size_b + j], tmp, 1e-5);
    }
  }
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

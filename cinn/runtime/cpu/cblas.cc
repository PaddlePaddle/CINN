#include "cinn/runtime/cpu/cblas.h"

#include <vector>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/common/cas.h"
#include "cinn/hlir/instruction/lower_impl.h"

namespace {

inline CBLAS_TRANSPOSE ToCblasTranspose(bool trans) { return trans ? CblasTrans : CblasNoTrans; }

}  // namespace

void cinn_cpu_mkl_gemm_fp32(float alpha,
                            int M,
                            int N,
                            int K,
                            bool ta,
                            bool tb,
                            int lda,
                            int ldb,
                            int ldc,
                            float beta,
                            cinn_buffer_t* A,
                            cinn_buffer_t* B,
                            cinn_buffer_t* C) {
  cblas_sgemm(CblasColMajor,
              ToCblasTranspose(ta),
              ToCblasTranspose(tb),
              M,
              N,
              K,
              alpha,
              reinterpret_cast<float*>(A->host_memory),
              lda,
              reinterpret_cast<float*>(B->host_memory),
              ldb,
              beta,
              reinterpret_cast<float*>(C->host_memory),
              ldc);
}

REGISTER_EXTERN_FUNC(cinn_cpu_mkl_gemm_fp32) {
  using namespace cinn;  // NOLINT
  using backends::FunctionProto;
  auto host_target = common::DefaultHostTarget();

  FunctionProto::shape_inference_t inference_shape = [](const std::vector<Expr>& args, int offset) {
    CHECK_EQ(offset, 0UL) << "Only one output";
    CHECK_EQ(args.size(), 12UL) << "Wrong number of arguments passed in";
    auto& A = args[10];
    auto& B = args[11];

    auto A_tensor = A.as_tensor();
    auto B_tensor = B.as_tensor();

    CHECK(A_tensor);
    CHECK(B_tensor);

    auto lda        = common::AutoSimplify(args[6]);
    int32_t lda_val = lda.as_int32();

    auto N = args[2];

    std::vector<Expr> shape;
    int total = 1;
    for (auto& v : A_tensor->shape) {
      auto val = common::AutoSimplify(v);
      CHECK(val.is_constant());
      shape.push_back(val);
      total *= val.as_int32();
      if (total >= lda_val) break;
    }

    shape.push_back(N);

    return shape;
  };

  REGISTER_EXTERN_FUNC_HELPER(cinn_cpu_mkl_gemm_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<float>()            // alpha
      .AddInputType<int>()              // M
      .AddInputType<int>()              // N
      .AddInputType<int>()              // K
      .AddInputType<bool>()             // ta
      .AddInputType<bool>()             // tb
      .AddInputType<int>()              // lda
      .AddInputType<int>()              // ldb
      .AddInputType<int>()              // ldc
      .AddInputType<float>()            // beta
      .AddInputType<cinn_buffer_t*>()   // A
      .AddInputType<cinn_buffer_t*>()   // B
      .AddOutputType<cinn_buffer_t*>()  // C
      .SetShapeInference(inference_shape)
      .End();
}

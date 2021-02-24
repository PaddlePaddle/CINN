#include "cinn/hlir/pe/transform.h"

#include <algorithm>
#include <utility>

#include "cinn/common/cas.h"
#include "cinn/common/context.h"
#include "cinn/common/ir_util.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using cinn::lang::Compute;
using ir::Tensor;

std::vector<Tensor> Matmul(
    const Tensor& A, const Tensor& B, bool trans_a, bool trans_b, float alpha, const std::string& name) {
  std::vector<Expr> shape_A = A->shape;
  std::vector<Expr> shape_B = B->shape;
  int a_dim                 = shape_A.size();
  int b_dim                 = shape_B.size();
  CHECK(a_dim == 3U || a_dim == 2U) << "tensor_A's dim should be 2 or 3 while current dim is " << a_dim;
  CHECK(b_dim == 3U || b_dim == 2U) << "tensor_B's dim should be 2 or 3 while current dim is " << b_dim;
  CHECK_EQ(a_dim, b_dim) << "tensor_A's dim should be same with tensor_B";

  Expr x_width  = trans_a ? shape_A[a_dim - 2] : shape_A.back();
  Expr y_height = trans_b ? shape_B.back() : shape_B[b_dim - 2];
  Expr M        = trans_a ? shape_A.back() : shape_A[a_dim - 2];
  Expr N        = trans_b ? shape_B[b_dim - 2] : shape_B.back();
  CHECK(is_zero(x_width - y_height)) << "matrix multiplication requires x_width to be same with y_height";
  std::vector<Expr> output_shape;
  std::vector<ir::Tensor> out;
  if (a_dim == 3) {
    int max_batch = std::max(shape_A[0].as_int32(), shape_B[0].as_int32());
    output_shape  = {Expr(max_batch), M, N};
  } else {
    output_shape = {M, N};
  }
  Var reduce_k(x_width, UniqName("reduce_k"));
  auto temp = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        int out_dim = indice.size();
        std::vector<Expr> A_indice;
        std::vector<Expr> B_indice;
        CHECK(out_dim == 3U || out_dim == 2U) << "indice size should be 2 or 3 while current dim is " << out_dim;
        if (out_dim == 3U) {
          // batch
          A_indice.push_back(indice[0]);
          B_indice.push_back(indice[0]);
        }
        A_indice.push_back(indice[out_dim - 2]);
        A_indice.push_back(reduce_k);
        B_indice.push_back(reduce_k);
        B_indice.push_back(indice[out_dim - 1]);
        if (trans_a) {
          std::swap(A_indice[out_dim - 2], A_indice[out_dim - 1]);
        }
        if (trans_b) {
          std::swap(B_indice[out_dim - 2], B_indice[out_dim - 1]);
        }
        return lang::ReduceSum(A(A_indice) * B(B_indice), {reduce_k});
      },
      "temp_matmul_out");
  if (alpha != 1) {
    auto res = Compute(
        output_shape,
        [=](const std::vector<Expr>& indice) { return temp(indice) * make_const(temp->type(), alpha); },
        name);
    return {res, temp};
  } else {
    return {temp};
  }
}

std::vector<Tensor> MatmulV2(const Tensor& A,
                             const Tensor& B,
                             bool trans_a,
                             bool trans_b,
                             float alpha,
                             const std::string& name,
                             const common::Target& target) {
  std::vector<Expr> shape_A = A->shape;
  std::vector<Expr> shape_B = B->shape;
  int a_dim                 = shape_A.size();
  int b_dim                 = shape_B.size();
  CHECK(a_dim == 3U || a_dim == 2U) << "tensor_A's dim should be 2 or 3 while current dim is " << a_dim;
  CHECK(b_dim == 3U || b_dim == 2U) << "tensor_B's dim should be 2 or 3 while current dim is " << b_dim;
  CHECK_EQ(a_dim, b_dim) << "tensor_A's dim should be same with tensor_B";

  Expr x_width  = trans_a ? shape_A[a_dim - 2] : shape_A.back();
  Expr y_height = trans_b ? shape_B.back() : shape_B[b_dim - 2];
  Expr M        = trans_a ? shape_A.back() : shape_A[a_dim - 2];
  Expr N        = trans_b ? shape_B[b_dim - 2] : shape_B.back();
  CHECK(is_zero(x_width - y_height)) << "matrix multiplication requires x_width to be same with y_height";
  Var reduce_k(x_width, UniqName("reduce_k"));
  std::vector<Expr> output_shape;
  std::vector<ir::Tensor> out;

  if (a_dim == 3) {
    int max_batch = std::max(shape_A[0].as_int32(), shape_B[0].as_int32());
    output_shape  = {Expr(max_batch), M, N};
  } else {
    output_shape = {M, N};
  }
  // array packing
  int shape_B_N = N.as_int32();
  int bn        = GetArrayPackingFactor(shape_B_N, B->type(), target);
  // {N / bn, K, bn}
  std::vector<Expr> packedB_shape = {Expr(shape_B_N / bn), y_height, Expr(bn)};
  if (b_dim == 3) {
    packedB_shape.insert(packedB_shape.begin(), output_shape[0]);
  }
  auto packedB = Compute(
      packedB_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> indice_b;
        int indice_dim = indice.size();
        CHECK_GE(indice_dim, 3) << "packedB's dim should be at least 3 while current dim is " << indice_dim;
        if (indice_dim == 4) {
          // batch
          indice_b.push_back(indice[0]);
        }
        // k
        indice_b.push_back(indice[indice_dim - 2]);
        indice_b.push_back(Expr(bn) * indice[indice_dim - 3] + indice.back());
        if (trans_b) {
          std::swap(indice_b.back(), indice_b[indice_b.size() - 2]);
        }
        return B(indice_b);
      },
      UniqName("packedB"));
  auto temp_res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> indice_a;
        std::vector<Expr> indice_b;
        int out_dim = indice.size();
        CHECK(out_dim == 3U || out_dim == 2U) << "indice size should be 2 or 3 while current dim is " << out_dim;
        if (out_dim == 3) {
          // batch
          indice_a.push_back(indice[0]);
          indice_b.push_back(indice[0]);
        }
        indice_a.push_back(indice[out_dim - 2]);
        indice_a.push_back(reduce_k);
        indice_b.push_back(indice[out_dim - 1] / Expr(bn));
        indice_b.push_back(reduce_k);
        indice_b.push_back(indice[out_dim - 1] % Expr(bn));
        if (trans_a) {
          std::swap(indice_a.back(), indice_a[indice_a.size() - 2]);
        }
        return lang::ReduceSum(A(indice_a) * packedB(indice_b), {reduce_k});
      },
      UniqName("temp_matmulV2_out"));
  if (alpha != 1) {
    auto res = Compute(
        output_shape,
        [=](const std::vector<Expr>& indice) { return temp_res(indice) * make_const(temp_res->type(), alpha); },
        name);
    return {res, temp_res, packedB};
  }
  return {temp_res, packedB};
}

std::vector<Tensor> MatmulMKL(const Tensor& A,
                              const Tensor& B,
                              bool trans_a,
                              bool trans_b,
                              float alpha,
                              const std::string& name,
                              const common::Target& target) {
  CHECK(target.arch == Target::Arch::X86) << "mkl should be used in the cpu environment";
  std::vector<Expr> shape_A = A->shape;
  std::vector<Expr> shape_B = B->shape;
  int a_dim                 = shape_A.size();
  int b_dim                 = shape_B.size();
  CHECK(a_dim == 3U || a_dim == 2U) << "tensor_A's dim should be 2 or 3 while current dim is " << a_dim;
  CHECK(b_dim == 3U || b_dim == 2U) << "tensor_B's dim should be 2 or 3 while current dim is " << b_dim;
  CHECK_EQ(a_dim, b_dim) << "tensor_A's dim should be same with tensor_B";
  if (a_dim == 3U) {
    CHECK_EQ(shape_A.front(), shape_B.front())
        << "tensor A and B's batch size should be same but current batch sizes are " << shape_A.front() << " and "
        << shape_B.front();
  }

  Expr x_width  = trans_a ? shape_A[a_dim - 2] : shape_A.back();
  Expr y_height = trans_b ? shape_B.back() : shape_B[b_dim - 2];
  Expr M        = trans_a ? shape_A.back() : shape_A[a_dim - 2];
  Expr N        = trans_b ? shape_B[b_dim - 2] : shape_B.back();
  CHECK(is_zero(x_width - y_height)) << "matrix multiplication requires x_width to be same with y_height";

  ir::Tensor call;
  if (a_dim == 2U) {
    call = Compute(
        {Expr(1)},
        [=]() -> Expr {
          return lang::CallExtern("cinn_cpu_mkl_gemm_fp32",
                                  {
                                      Expr(alpha),                 // alpha
                                      M,                           // M
                                      N,                           // N
                                      x_width,                     // K
                                      common::make_bool(trans_a),  // ta
                                      common::make_bool(trans_b),  // tb
                                      shape_A.back(),              // lda
                                      shape_B.back(),              // ldb
                                      N,                           // ldc
                                      common::make_zero<float>(),  // beta
                                      A,                           // A
                                      B,                           // B
                                  });
        },
        UniqName("matmul_mkl_out"));
  } else {
    // batch matmul
    call = Compute(
        {Expr(1)},
        [=]() -> Expr {
          return lang::CallExtern("cinn_cpu_mkl_gemm_batch_fp32",
                                  {
                                      Expr(alpha),                 // alpha
                                      shape_A.front(),             // batch
                                      M,                           // M
                                      N,                           // N
                                      x_width,                     // K
                                      common::make_bool(trans_a),  // ta
                                      common::make_bool(trans_b),  // tb
                                      shape_A.back(),              // lda
                                      shape_B.back(),              // ldb
                                      N,                           // ldc
                                      M * x_width,                 // a_stride
                                      N * x_width,                 // b_stride
                                      M * N,                       // c_stride
                                      common::make_zero<float>(),  // beta
                                      A,                           // A
                                      B,                           // B
                                  });
        },
        UniqName("batch_matmul_mkl_out"));
  }
  auto out = call->TupleGet(0);
  out->WithBuffer(A->type());
  return {out, call};
}

int GetMulFactor(int shape, const Type& type, const common::Target& target) {
  int split_base   = GetBasicFactor(type, target);
  int split_factor = 1;
  for (size_t i = split_base; i >= 1; --i) {
    if (shape % i == 0) {
      split_factor = i;
      break;
    }
  }
  return split_factor;
}

std::vector<Tensor> MulBase(const Tensor& A, const Tensor& B, const std::string& name, const common::Target& target) {
  std::vector<Expr> output_shape;
  CHECK_EQ(A->shape.size(), 2U) << "tensor_A's shape size should be two while current shape size is "
                                << A->shape.size();
  CHECK_EQ(B->shape.size(), 2U) << "tensor_B's shape size should be two while current shape size is "
                                << B->shape.size();
  CHECK_EQ(A->shape[1], B->shape[1]) << "tensor_A's last shape should be same with tensor_B";
  output_shape.push_back(A->shape[0]);
  output_shape.push_back(B->shape[0]);

  if (target.arch == Target::Arch::X86) {
    int reduce_dim   = A->shape[1].as_int32();
    int split_factor = GetMulFactor(reduce_dim, A->type(), target);
    Var reduce_k_first(common::make_const(A->shape[1]->type(), reduce_dim / split_factor), UniqName("reduce_k_first"));
    auto mul_reduce_first = Compute(
        {A->shape[0], B->shape[0], Expr(split_factor)},
        [=](const std::vector<Expr>& indice) {
          CHECK_EQ(indice.size(), 3U) << "indice size should be three while current size is " << indice.size();
          return lang::ReduceSum(A({indice[0], reduce_k_first * Expr(split_factor) + indice[2]}) *
                                     B({indice[1], reduce_k_first * Expr(split_factor) + indice[2]}),
                                 {reduce_k_first});
        },
        UniqName("mul_reduce_k_first"));
    Var reduce_k_second(common::make_const(A->shape[1]->type(), split_factor), UniqName("reduce_k_second"));
    return {Compute(
                output_shape,
                [=](const std::vector<Expr>& indice) {
                  std::vector<Expr> new_indice = indice;
                  new_indice.push_back(reduce_k_second);
                  return lang::ReduceSum(mul_reduce_first(new_indice), {reduce_k_second});
                },
                name),
            mul_reduce_first};
  } else {
    Var reduce_k(A->shape[1], UniqName("reduce_k"));
    return {Compute(
        output_shape,
        [=](const std::vector<Expr>& indice) {
          std::vector<Expr> A_indice;
          std::vector<Expr> B_indice;
          CHECK_EQ(indice.size(), 2U) << "indice size should be two while current size is " << indice.size();
          A_indice.push_back(indice[0]);
          B_indice.push_back(indice[1]);
          A_indice.push_back(reduce_k);
          B_indice.push_back(reduce_k);
          return lang::ReduceSum(A(A_indice) * B(B_indice), {reduce_k});
        },
        name)};
  }
}

std::vector<Tensor> Mul(const Tensor& A,
                        const Tensor& B,
                        int x_num_col_dims,
                        const std::vector<Expr>& output_shape,
                        const Var& axis_k,
                        const std::string& name) {
  return {Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> A_indice;
        std::vector<Expr> B_indice;
        A_indice.insert(A_indice.begin(), indice.begin(), indice.begin() + x_num_col_dims);
        B_indice.insert(B_indice.begin(), indice.begin() + x_num_col_dims, indice.end());
        A_indice.push_back(axis_k);
        B_indice.push_back(axis_k);
        return lang::ReduceSum(A(A_indice) * B(B_indice), {axis_k});
      },
      name)};
}

std::vector<Tensor> MulMKL(const Tensor& A, const Tensor& B, const std::string& name, const common::Target& target) {
  CHECK(target.arch == Target::Arch::X86) << "mkl should be used in the cpu environment";
  std::vector<Expr> shape_A = A->shape;
  std::vector<Expr> shape_B = B->shape;
  int a_dim                 = shape_A.size();
  int b_dim                 = shape_B.size();
  CHECK_EQ(a_dim, 2U) << "tensor_A's shape size should be two while current shape size is " << A->shape.size();
  CHECK_EQ(b_dim, 2U) << "tensor_B's shape size should be two while current shape size is " << B->shape.size();
  // A: [M, K], B: [N, K]
  Expr x_width  = shape_A[1];
  Expr y_height = shape_B[1];
  Expr M        = shape_A[0];
  Expr N        = shape_B[0];
  CHECK(is_zero(x_width - y_height)) << "matrix multiplication requires x_width to be same with y_height";
  CHECK_EQ(A->shape[1], B->shape[1]) << "tensor_A's last shape should be same with tensor_B";

  auto call = Compute(
      {Expr(1)},
      [=]() -> Expr {
        return lang::CallExtern("cinn_cpu_mkl_gemm_fp32",
                                {
                                    common::make_const(Float(32), 1),  // alpha
                                    M,                                 // M
                                    N,                                 // N
                                    x_width,                           // K
                                    common::make_bool(false),          // ta
                                    common::make_bool(true),           // tb
                                    shape_A.back(),                    // lda
                                    shape_B.back(),                    // ldb
                                    N,                                 // ldc
                                    common::make_zero<float>(),        // beta
                                    A,                                 // A
                                    B,                                 // B
                                });
      },
      UniqName("mul_mkl_out"));
  auto out = call->TupleGet(0);
  out->WithBuffer(A->type());
  return {out, call};
}

std::vector<ir::Tensor> MulBias(const Tensor& A,
                                const Tensor& B,
                                const Tensor& C,
                                int x_num_col_dims,
                                const std::vector<Expr>& output_shape,
                                const Var& axis_k,
                                const std::string& name) {
  auto temp = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> A_indice;
        std::vector<Expr> B_indice;
        A_indice.insert(A_indice.begin(), indice.begin(), indice.begin() + x_num_col_dims);
        B_indice.insert(B_indice.begin(), indice.begin() + x_num_col_dims, indice.end());
        A_indice.push_back(axis_k);
        B_indice.push_back(axis_k);
        return lang::ReduceSum(A(A_indice) * B(B_indice), {axis_k});
      },
      UniqName("temp_out_mulbias"));
  auto res = Compute(
      output_shape, [=](const std::vector<Expr>& indice) { return temp(indice) + C(indice); }, name);
  return {temp, res};
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/hlir/pe/transform.h"

#include <algorithm>
#include <utility>

#include "cinn/common/cas.h"
#include "cinn/common/context.h"
#include "cinn/common/ir_util.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/utils/string.h"

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

ir::Tensor Reshape(const ir::Tensor& A,
                   const std::vector<int>& new_shape,
                   poly::StageMap stages,
                   const std::string& name) {
  std::vector<Expr> new_expr_shape;
  std::vector<Expr> A_expr_shape = A->shape;
  int input_total_size           = 1;
  int output_total_size          = 1;
  for (auto& i : A_expr_shape) {
    CHECK(i.is_constant()) << "Input tensor's shape should be constant value.";
    input_total_size *= static_cast<int>(i.get_constant());
  }
  for (auto& i : new_shape) {
    output_total_size *= i;
    new_expr_shape.push_back(Expr(i));
  }
  CHECK_EQ(input_total_size, output_total_size)
      << "In op reshape, the input tensor and output tensor's total size should be equal, please check!";
  auto out = Identity(A->Reshape(new_expr_shape, stages), name).front();
  return out;
}

ir::Tensor Reshape(const ir::Tensor& A, const std::vector<int>& new_shape, const std::string& name) {
  std::vector<Expr> new_expr_shape;
  std::vector<Expr> A_expr_shape = A->shape;
  int input_total_size           = 1;
  int output_total_size          = 1;
  for (auto& i : A_expr_shape) {
    CHECK(i.is_constant()) << "Input tensor's shape should be constant value.";
    input_total_size *= static_cast<int>(i.get_constant());
  }
  for (auto& i : new_shape) {
    output_total_size *= i;
    new_expr_shape.push_back(Expr(i));
  }
  CHECK_EQ(input_total_size, output_total_size)
      << "In op reshape, the input tensor and output tensor's total size should be equal, please check!";
  auto res = Compute(
      new_expr_shape,
      [=](const std::vector<Expr>& indice) {
        Expr offset = Expr(0);
        for (int i = 0; i < indice.size(); i++) {
          offset = offset * new_expr_shape[i] + indice[i];
        }
        std::vector<Expr> indice_a;
        for (int i = A_expr_shape.size() - 1; i >= 0; i--) {
          auto temp = offset % A_expr_shape[i];
          indice_a.insert(indice_a.begin(), common::AutoSimplify(temp));
          offset = (offset - temp) / A_expr_shape[i];
        }
        return A(indice_a);
      },
      name);
  return res;
}

ir::Tensor Concat(const ir::Tensor& A, const ir::Tensor& B, int axis, const std::string& name) {
  if (axis < 0) axis += A->shape.size();
  CHECK_EQ(A->shape.size(), B->shape.size()) << "Dimensions of inputs A and B in Concat should be equal! Please check.";
  std::vector<Expr> output_shape = A->shape;
  Expr pivot                     = A->shape[axis];
  output_shape[axis]             = common::AutoSimplify(output_shape[axis] + B->shape[axis]);
  auto res                       = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        auto indice_B  = indice;
        indice_B[axis] = indice_B[axis] - pivot;
        return ir::Select::Make(indice[axis] < pivot, A(indice), B(indice_B));
      },
      name);
  return res;
}

ir::Tensor Concat(const std::vector<ir::Tensor>& input_tensors, int axis, const std::string& name) {
  int input_size = input_tensors.size();
  CHECK_GE(input_size, 2U) << "Concat should have at least 2 input tensors";
  std::vector<Expr> output_shape = input_tensors[0]->shape;
  int input_dim                  = output_shape.size();
  CHECK(axis >= -input_dim && axis < input_dim) << "Concat's axis should be in [-R, R)"
                                                << ", but get axis: " << axis << ", R: " << input_dim;
  if (axis < 0) axis += output_shape.size();

  for (int i = 1; i < input_size; i++) {
    CHECK_EQ(input_tensors[i]->shape.size(), input_dim)
        << "Dimensions of inputs tensors in Concat should be equal! Please check.";
    output_shape[axis] = common::AutoSimplify(output_shape[axis] + input_tensors[i]->shape[axis]);
  }

  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        auto ret              = input_tensors[0](indice);
        Expr accumulate_shape = Expr(0);
        for (int i = 0; i < input_size - 1; i++) {
          accumulate_shape             = common::AutoSimplify(accumulate_shape + input_tensors[i]->shape[axis]);
          std::vector<Expr> new_indice = indice;
          new_indice[axis]             = indice[axis] - accumulate_shape;
          ret = ir::Select::Make(indice[axis] < accumulate_shape, ret, input_tensors[i + 1](new_indice));
        }
        return ret;
      },
      name);
  return res;
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

  auto res = Compute(
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
        if (alpha == 1) {
          return lang::ReduceSum(A(indice_a) * packedB(indice_b), {reduce_k});
        } else {
          return lang::ReduceSum(A(indice_a) * packedB(indice_b) * make_const(A->type(), alpha), {reduce_k});
        }
      },
      UniqName("matmulV2_out"));
  return {res, packedB};
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

void GetLayoutTransformInfo(const ir::Layout& src_layout,
                            const ir::Layout& dst_layout,
                            absl::flat_hash_map<int, std::vector<int>>* split_index_map) {
  CHECK_GT(dst_layout.ndims(), src_layout.ndims());
  int offset = 'A' - 'a';
  CHECK_EQ(dst_layout.axis_names().size(), dst_layout.ndims());
  for (int i = dst_layout.ndims() - 1; i >= 0; i--) {
    char axis_name      = dst_layout.axis_names(i);
    char prim_axis_name = axis_name;
    if (axis_name >= 'a' && axis_name <= 'z') {
      prim_axis_name += offset;
      int factor = dst_layout[i]->upper_bound.as_int32();

      CHECK_GT(factor, 0) << "sub-axis factor should be larger than 0";
      int src_primal_index = src_layout.axis_names().find(prim_axis_name);
      int dst_primal_index = dst_layout.axis_names().find(prim_axis_name);
      CHECK(src_primal_index != src_layout.axis_names().npos);
      CHECK(dst_primal_index != dst_layout.axis_names().npos);
      (*split_index_map)[src_primal_index] = {dst_primal_index, i, factor};
    } else {
      int src_primal_index = src_layout.axis_names().find(prim_axis_name);
      if (split_index_map->find(src_primal_index) != split_index_map->end()) continue;
      CHECK(src_primal_index != src_layout.axis_names().npos);
      (*split_index_map)[src_primal_index] = {i};
    }
  }
}

std::vector<Expr> InferShapeLayoutTransform(const std::vector<Expr>& input_shapes,
                                            const ir::Layout& old_layout,
                                            const ir::Layout& new_layout,
                                            absl::flat_hash_map<int, std::vector<int>>* split_index_map) {
  int src_dim = old_layout.ndims();
  int dst_dim = new_layout.ndims();
  std::vector<Expr> output_shape(dst_dim);
  CHECK_EQ(input_shapes.size(), src_dim);

  if (src_dim == dst_dim) {
    CHECK_EQ(old_layout.name(), new_layout.name());
    return input_shapes;
  } else if (src_dim < dst_dim) {
    GetLayoutTransformInfo(old_layout, new_layout, split_index_map);
    for (int i = 0; i < src_dim; i++) {
      CHECK(split_index_map->find(i) != split_index_map->end());
      if ((*split_index_map)[i].size() == 3) {
        int dst_prim_index           = (*split_index_map)[i][0];
        int dst_sub_index            = (*split_index_map)[i][1];
        int factor                   = (*split_index_map)[i][2];
        Expr chunk_shape             = common::AutoSimplify(input_shapes[i] / factor);
        Expr block_shape             = Expr(factor);
        output_shape[dst_prim_index] = chunk_shape;
        output_shape[dst_sub_index]  = block_shape;
      } else if ((*split_index_map)[i].size() == 1) {
        int dst_prim_index           = (*split_index_map)[i][0];
        output_shape[dst_prim_index] = input_shapes[i];
      }
    }
  } else {
    GetLayoutTransformInfo(new_layout, old_layout, split_index_map);
    for (int i = 0; i < dst_dim; i++) {
      CHECK(split_index_map->find(i) != split_index_map->end());
      if ((*split_index_map)[i].size() == 3) {
        int src_prim_index = (*split_index_map)[i][0];
        int src_sub_index  = (*split_index_map)[i][1];
        int factor         = (*split_index_map)[i][2];
        CHECK_GE(input_shapes.size(), src_sub_index);
        CHECK_EQ(input_shapes[src_sub_index].as_int32(), factor);
        output_shape[i] = common::AutoSimplify(input_shapes[src_prim_index] * factor);
      } else if ((*split_index_map)[i].size() == 1) {
        int src_prim_index = (*split_index_map)[i][0];
        output_shape[i]    = input_shapes[src_prim_index];
      }
    }
  }
  VLOG(4) << "output_shape: " << output_shape;
  return output_shape;
}

ir::Tensor LayoutTransform(const Tensor& input,
                           const std::string& src_layout,
                           const std::string& dst_layout,
                           const std::string& name) {
  CHECK(src_layout != dst_layout) << "dst_layout is same with src_layout, should not do layout transform";
  // NCHW -> NCHWxc
  // NCHWxc -> NCHW
  // OIHW -> OIHWxixo
  // OIHWxixo -> OIHW
  CHECK_GE(src_layout.size(), 4U);
  CHECK_GE(dst_layout.size(), 4U);
  absl::flat_hash_map<int, std::vector<int>> split_index_map;
  // transform shape
  int offset = 'A' - 'a';
  ir::Layout old_layout(src_layout);
  ir::Layout new_layout(dst_layout);
  int src_dim                    = old_layout.ndims();
  int dst_dim                    = new_layout.ndims();
  std::vector<Expr> output_shape = InferShapeLayoutTransform(input->shape, old_layout, new_layout, &split_index_map);
  CHECK_EQ(output_shape.size(), dst_dim);

  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        // transform indice
        std::vector<Expr> new_indice(src_dim);
        int min_dim = std::min(src_dim, dst_dim);
        for (int i = 0; i < min_dim; i++) {
          CHECK(split_index_map.find(i) != split_index_map.end());
          std::vector<int> split_infos = split_index_map.at(i);
          if (split_infos.size() == 3) {
            int prim_index = split_infos[0];
            int sub_index  = split_infos[1];
            int factor     = split_infos[2];
            if (dst_dim > src_dim) {
              new_indice[i] = common::AutoSimplify(indice[prim_index] * factor + indice[sub_index]);
            } else {
              new_indice[prim_index] = common::AutoSimplify(indice[i] / factor);
              new_indice[sub_index]  = common::AutoSimplify(indice[i] % factor);
            }

          } else if (split_infos.size() == 1) {
            int prim_index = split_infos[0];
            if (dst_dim > src_dim) {
              new_indice[i] = indice[prim_index];
            } else {
              new_indice[prim_index] = indice[i];
            }
          }
        }
        VLOG(4) << "new_indice: " << new_indice;

        return input(new_indice);
      },
      name);
  return {res};
}

ir::Tensor Reverse(const ir::Tensor& input, const std::vector<int>& axis, const std::string& output_name) {
  for (auto& val : axis) {
    CHECK(val >= 0 && val < static_cast<int>(input->shape.size())) << "axis should be [0,n_dim)";
  }
  std::vector<Expr> shape = input->shape;
  return lang::Compute(
      input->shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> indexs(indice.begin(), indice.end());
        for (auto idx : axis) {
          indexs[idx] = shape[idx] - Expr(1) - indexs[idx];
        }
        return input(indexs);
      },
      output_name);
}

ir::Tensor Transpose(const ir::Tensor& input, const std::vector<int>& axis, const std::string& output_name) {
  CHECK_EQ(input->shape.size(), axis.size()) << "input shape size and axis size is not equal!";
  for (int idx = 0; idx < axis.size(); ++idx) {
    CHECK(axis[idx] >= 0 && axis[idx] < axis.size()) << "axis value should be among [0,axis.size())";
    for (int idy = idx + 1; idy < axis.size(); ++idy) {
      CHECK_NE(axis[idx], axis[idy]) << "axis value can't repeat!";
    }
  }
  // compute output shape
  std::vector<Expr> shape = input->shape;
  std::vector<Expr> output_shape;
  for (auto idx = 0; idx < axis.size(); ++idx) {
    output_shape.push_back(shape[axis[idx]]);
  }

  // tranpose axis to map output to input
  // new_axis = axis(T)
  std::vector<int> new_axis;
  for (int idx = 0; idx < axis.size(); ++idx) {
    for (int idy = 0; idy < axis.size(); ++idy) {
      if (idx == axis[idy]) {
        new_axis.push_back(idy);
      }
    }
  }

  return lang::Compute(
      output_shape,
      [=](const std::vector<Expr>& indice) {
        std::vector<Expr> indexs;
        for (auto idx : new_axis) {
          indexs.push_back(indice[idx]);
        }
        return input(indexs);
      },
      output_name);
}

std::vector<ir::Tensor> IndexSelect(const ir::Tensor& x,
                                    const ir::Tensor& index,
                                    const std::vector<Expr>& output_shape,
                                    int axis,
                                    const std::string& name) {
  int outer_num = 1;
  for (auto i = 0; i < axis; i++) {
    outer_num *= x->shape[i].as_int32();
  }
  VLOG(1) << "The outer_num calculated in pe::IndexSelect = " << outer_num;

  int slice_size = 1;
  for (int i = axis + 1; i < x->shape.size(); i++) {
    slice_size *= x->shape[i].as_int32();
  }
  VLOG(1) << "The slice_size calculated in pe::IndexSelect = " << slice_size;

  auto reshape_input = pe::Reshape(x, {outer_num, x->shape[axis].as_int32(), slice_size}, name + "_reshape");

  int index_size = index->shape[0].as_int32();
  std::vector<ir::Tensor> output_one_tensors(index_size);
  std::vector<Expr> output_one_shape = output_shape;
  output_one_shape[axis]             = Expr(1);
  for (int i = 0; i < index_size; i++) {
    auto output_one = Compute(
        output_one_shape,
        [reshape_input, index, axis, i](const std::vector<Expr>& indice) {
          Expr index_val                   = ir::Cast::Make(common::Int(32), index(Expr(i)));
          std::vector<Expr> mutable_indice = indice;
          mutable_indice[axis]             = mutable_indice[axis] + index_val;
          return reshape_input(mutable_indice);
        },
        name + std::to_string(i));
    output_one_tensors[i] = std::move(output_one);
  }

  auto concat_output = pe::Concat(output_one_tensors, axis, name + "_concat");
  std::vector<int> output_shape_val(output_shape.size());
  std::transform(output_shape.begin(), output_shape.end(), output_shape_val.begin(), [](const Expr& expr) {
    return expr.as_int32();
  });
  VLOG(1) << "The output shape used in IndexSelect: " << utils::Join(output_shape_val, ", ");

  auto reshape_output = pe::Reshape(concat_output, output_shape_val, name);

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(index_size + 3);
  output_tensors.emplace_back(std::move(reshape_output));
  output_tensors.emplace_back(std::move(concat_output));
  output_tensors.emplace_back(std::move(reshape_input));
  output_tensors.insert(output_tensors.end(), output_one_tensors.begin(), output_one_tensors.end());
  return output_tensors;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

#include "cinn/hlir/pe/transform.h"

#include <algorithm>

#include "cinn/common/cas.h"
#include "cinn/common/context.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using ir::Tensor;
using cinn::lang::Compute;
void GetMatmulOutputShape(const std::vector<Expr>& shape1,
                          const std::vector<Expr>& shape2,
                          std::vector<Expr>* shape1_new,
                          std::vector<Expr>* shape2_new,
                          std::vector<Expr>* output_shape,
                          bool trans_a,
                          bool trans_b,
                          int x_num_col_dims,
                          int y_num_col_dims) {
  CHECK(shape1_new);
  CHECK(shape2_new);
  CHECK(output_shape);
  *shape1_new = shape1;
  *shape2_new = shape2;
  if (trans_a) {
    std::reverse(shape1_new->begin(), shape1_new->end());
  }
  if (trans_b) {
    std::reverse(shape2_new->begin(), shape2_new->end());
  }
  // first get output shape
  output_shape->insert(output_shape->begin(), shape1_new->begin(), shape1_new->begin() + x_num_col_dims);
  output_shape->insert(output_shape->end(), shape2_new->begin() + y_num_col_dims, shape2_new->end());
}

void GetMatmulIndice(const std::vector<Expr>& shape1_new,
                     const std::vector<Expr>& shape2_new,
                     const std::vector<Expr>& indices,
                     bool trans_a,
                     bool trans_b,
                     int x_num_col_dims,
                     int y_num_col_dims,
                     std::vector<Expr>* indice1,
                     std::vector<Expr>* indice2,
                     std::vector<Var>* reduce_axes) {
  CHECK(indice1);
  CHECK(indice2);
  CHECK(reduce_axes);
  if (indice1->empty() && indice2->empty()) {
    CHECK_GE(indices.size(), x_num_col_dims);
    for (size_t i = 0; i < x_num_col_dims; i++) {
      indice1->emplace_back(indices[i]);
    }
    Expr reduce_shape1 = Expr(1);
    // A reduce axes
    for (size_t i = x_num_col_dims; i < shape1_new.size(); i++) {
      reduce_shape1           = reduce_shape1 * shape1_new[i];
      std::string reduce_name = UniqName("kk");
      auto k                  = Var(shape1_new[i], reduce_name);
      reduce_axes->emplace_back(k);
      indice1->emplace_back(k);
    }
    Expr reduce_shape2 = Expr(1);
    // B reduce axes
    for (size_t i = 0; i < y_num_col_dims; i++) {
      reduce_shape2 = reduce_shape2 * shape2_new[i];
      reduce_shape2 = common::AutoSimplify(reduce_shape2);
      indice2->emplace_back((*indice1)[indice1->size() - 1 - i]);
    }

    CHECK(MathEqual(reduce_shape1, reduce_shape2))
        << "reduce shape not match: " << reduce_shape1 << " vs " << reduce_shape2;
    CHECK_GE(indices.size(), shape2_new.size() - y_num_col_dims);
    for (size_t i = y_num_col_dims; i < shape2_new.size(); i++) {
      indice2->emplace_back(indices[x_num_col_dims + i - y_num_col_dims]);
    }
    if (trans_a) {
      std::reverse(indice1->begin(), indice1->end());
    }
    if (trans_b) {
      std::reverse(indice2->begin(), indice2->end());
    }
  }
}

Tensor Matmul(const Tensor& A,
              const Tensor& B,
              bool trans_a,
              bool trans_b,
              int x_num_col_dims,
              int y_num_col_dims,
              const std::string& name) {
  std::vector<Expr> output_shape;
  std::vector<Expr> shape1_new;
  std::vector<Expr> shape2_new;
  std::vector<Expr> A_indice;
  std::vector<Expr> B_indice;
  std::vector<Var> reduce_axes;
  GetMatmulOutputShape(
      A->shape, B->shape, &shape1_new, &shape2_new, &output_shape, trans_a, trans_b, x_num_col_dims, y_num_col_dims);

  auto fn = [&](const std::vector<Expr>& indices) {
    GetMatmulIndice(shape1_new,
                    shape2_new,
                    indices,
                    trans_a,
                    trans_b,
                    x_num_col_dims,
                    y_num_col_dims,
                    &A_indice,
                    &B_indice,
                    &reduce_axes);
    return lang::ReduceSum(A(A_indice) * B(B_indice), reduce_axes);
  };
  return Compute(output_shape, fn, name);
}

int GetMulReduceFactor(int reduce_shape, const Type& type, const common::Target& target) {
  int target_native_vector_bits = target.get_target_bits() * 8;
  int type_bits                 = type.bits();
  int split_base                = target_native_vector_bits / type_bits;
  int split_factor              = 1;
  int reduce_dim                = reduce_shape;
  for (size_t i = split_base * 2; i >= 1; --i) {
    if (reduce_dim % i == 0) {
      split_factor = i;
      break;
    }
  }
  return split_factor;
}

std::vector<Tensor> MulBase(const Tensor& A, const Tensor& B, const std::string& name, const common::Target& target) {
  std::vector<Expr> output_shape;
  CHECK_EQ(A->shape.size(), 2U) << "tensor_A's shape size should be two";
  CHECK_EQ(B->shape.size(), 2U) << "tensor_B's shape size should be two";
  CHECK_EQ(A->shape[1], B->shape[1]) << "tensor_A's last shape should be same with tensor_B";
  output_shape.push_back(A->shape[0]);
  output_shape.push_back(B->shape[0]);

  if (target.arch == Target::Arch::X86) {
    int reduce_dim   = A->shape[1].as_int32();
    int split_factor = GetMulReduceFactor(reduce_dim, A->type(), target);
    Var reduce_k_first(common::make_const(A->shape[1]->type(), reduce_dim / split_factor), UniqName("reduce_k_first"));
    auto mul_reduce_first = Compute(
        {A->shape[0], B->shape[0], Expr(split_factor)},
        [=](const std::vector<Expr>& indice) {
          CHECK_EQ(indice.size(), 3U) << "indice size should be three";
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
          CHECK_EQ(indice.size(), 2U) << "indice size should be two";
          A_indice.push_back(indice[0]);
          B_indice.push_back(indice[1]);
          A_indice.push_back(reduce_k);
          B_indice.push_back(reduce_k);
          return lang::ReduceSum(A(A_indice) * B(B_indice), {reduce_k});
        },
        name)};
  }
}

Tensor Mul(const Tensor& A,
           const Tensor& B,
           int x_num_col_dims,
           const std::vector<Expr>& output_shape,
           const Var& axis_k,
           const std::string& name) {
  return Compute(
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
      name);
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

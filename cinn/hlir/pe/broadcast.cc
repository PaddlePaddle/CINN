#include "cinn/hlir/pe/broadcast.h"

#include <algorithm>
#include <vector>

#include "cinn/common/ir_util.h"
#include "cinn/ir/node.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using cinn::common::make_zero;
using cinn::ir::_Var_;
using cinn::ir::Max;
using cinn::ir::Tensor;
using cinn::lang::Compute;

void GetBroadcastShape(const std::vector<Expr>& shape1,
                       const std::vector<Expr>& shape2,
                       std::vector<Expr>* common_shape,
                       std::vector<bool>* broadcast_flag1,
                       std::vector<bool>* broadcast_flag2) {
  CHECK(common_shape && broadcast_flag1 && broadcast_flag2);
  int size1 = shape1.size();
  int size2 = shape2.size();
  Expr one(1);
  int i;
  for (i = 1; i <= std::min(size1, size2); ++i) {
    const _Var_* var1 = shape1[size1 - i].As<_Var_>();
    const _Var_* var2 = shape2[size2 - i].As<_Var_>();
    if (MathEqual(shape1[size1 - i], shape2[size2 - i])) {
      common_shape->insert(common_shape->begin(), shape1[size1 - i]);
      broadcast_flag1->insert(broadcast_flag1->begin(), true);
      broadcast_flag2->insert(broadcast_flag2->begin(), true);
    } else if (MathEqual(one, shape1[size1 - i])) {
      CHECK(!MathEqual(one, shape2[size2 - i]));
      common_shape->insert(common_shape->begin(), shape2[size2 - i]);
      broadcast_flag1->insert(broadcast_flag1->begin(), false);
      broadcast_flag2->insert(broadcast_flag2->begin(), true);
    } else if (MathEqual(one, shape2[size2 - i])) {
      CHECK(!MathEqual(one, shape1[size1 - i]));
      common_shape->insert(common_shape->begin(), shape1[size1 - i]);
      broadcast_flag1->insert(broadcast_flag1->begin(), true);
      broadcast_flag2->insert(broadcast_flag2->begin(), false);
    } else if (var1 && var2) {
      Expr max_var = Max::Make(shape1[size1 - i], shape2[size2 - i]);
      common_shape->insert(common_shape->begin(), max_var);
      broadcast_flag1->insert(broadcast_flag1->begin(), true);
      broadcast_flag2->insert(broadcast_flag2->begin(), true);
    } else if (var1) {
      common_shape->insert(common_shape->begin(), shape2[size2 - i]);
      broadcast_flag1->insert(broadcast_flag1->begin(), true);
      broadcast_flag2->insert(broadcast_flag2->begin(), true);
    } else if (var2) {
      common_shape->insert(common_shape->begin(), shape1[size1 - i]);
      broadcast_flag1->insert(broadcast_flag1->begin(), true);
      broadcast_flag2->insert(broadcast_flag2->begin(), true);
    } else {
      CHECK(false) << "Incompatible broadcast dims: " << shape1[size1 - i] << " and " << shape2[size2 - i]
                   << " in: " << shape1 << " and " << shape2 << std::endl;
    }
  }
  if (size1 != size2) {
    int max_size = std::max(size1, size2);
    auto& shape  = (size1 > size2) ? shape1 : shape2;
    auto var_l   = (size1 > size2) ? broadcast_flag1 : broadcast_flag2;
    for (; i <= max_size; ++i) {
      common_shape->insert(common_shape->begin(), shape[max_size - i]);
      var_l->insert(var_l->begin(), true);
    }
  }
}

void GetBroadcastIndice(const std::vector<Expr>& indice,
                        std::vector<Expr>* broadcast_indice1,
                        std::vector<Expr>* broadcast_indice2,
                        const std::vector<bool>& broadcast_flags1,
                        const std::vector<bool>& broadcast_flags2) {
  CHECK(broadcast_indice1 && broadcast_indice2);
  if (broadcast_indice1->empty() && broadcast_indice2->empty()) {
    int flag_size1 = broadcast_flags1.size();
    int flag_size2 = broadcast_flags2.size();
    int i;
    for (i = 0; i < std::min(flag_size1, flag_size2); i++) {
      auto& indice1 = flag_size1 ? indice[i] : make_zero();
      auto& indice2 = flag_size2 ? indice[i] : make_zero();
      broadcast_indice1->push_back(indice1);
      broadcast_indice2->push_back(indice2);
    }
    if (flag_size1 != flag_size2) {
      auto broadcast_indice_last = (flag_size1 > flag_size2) ? broadcast_indice1 : broadcast_indice2;
      for (; i < indice.size(); i++) {
        broadcast_indice_last->push_back(indice[i]);
      }
    }
  }
}

template <typename FuncOp>
Tensor Broadcast(const FuncOp& op, const Tensor& a, const Tensor& b, const std::string& output_name = "") {
  std::vector<Expr> common_shape;
  std::vector<bool> broadcast_flags1;
  std::vector<bool> broadcast_flags2;
  std::vector<Expr> broadcast_indice1;
  std::vector<Expr> broadcast_indice2;

  GetBroadcastShape(a->shape, b->shape, &common_shape, &broadcast_flags1, &broadcast_flags2);
  auto fn = [&](const std::vector<Expr>& indice) {
    GetBroadcastIndice(indice, &broadcast_indice1, &broadcast_indice2, broadcast_flags1, broadcast_flags2);
    return op(a(broadcast_indice1), b(broadcast_indice2));
  };
  Tensor output = Compute(common_shape, fn, output_name);
  return output;
}

/**
 * @brief Compute A && B with auto-broadcasting.
 *
 * @param A The first tensor or Expr
 * @param B The second tensor or Expr
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor or Expr.
 */
#define HLIR_DEFINE_BC_OP(name__, compute__)                                                       \
  Tensor name__(const Tensor& A, const Tensor& B, const std::string& output_name) {                \
    auto fn = [&](const Expr& a, const Expr& b) { compute__ };                                     \
    return Broadcast(fn, A, B, output_name);                                                       \
  }                                                                                                \
  Tensor name__(const Tensor& A, const Expr& B, const std::string& output_name) {                  \
    auto fn = [&](const Expr& a, const Expr& b) { compute__ };                                     \
    return Compute(                                                                                \
        A->shape, [&](const std::vector<Expr>& indice) { return fn(A(indice), B); }, output_name); \
  }                                                                                                \
  Tensor name__(const Expr& A, const Tensor& B, const std::string& output_name) {                  \
    auto fn = [&](const Expr& a, const Expr& b) { compute__ };                                     \
    return Compute(                                                                                \
        B->shape, [&](const std::vector<Expr>& indice) { return fn(A, B(indice)); }, output_name); \
  }                                                                                                \
  Expr name__(const Expr& a, const Expr& b) { compute__ }

HLIR_DEFINE_BC_OP(Add, return a + b;);
HLIR_DEFINE_BC_OP(Substract, return a - b;);
HLIR_DEFINE_BC_OP(Multiply, return a * b;);
HLIR_DEFINE_BC_OP(Divide, return a / b;);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

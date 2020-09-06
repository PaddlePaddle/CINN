#include "cinn/hlir/pe/broadcast.h"

#include <vector>

#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/node.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using namespace cinn::ir;
using cinn::common::make_zero;
using cinn::lang::Compute;

void GetBroadcastShape(const std::vector<Expr>& shape1,
                       const std::vector<Expr>& shape2,
                       std::vector<Expr>* common_shape,
                       std::vector<bool>* broadcast_flag1,
                       std::vector<bool>* broadcast_flag2,
                       const Expr& axis) {
  CHECK(common_shape);
  CHECK(broadcast_flag1);
  CHECK(broadcast_flag2);
  std::vector<Expr> shape2_new = shape2;
  if (axis.defined()) {
    int axis_val = axis.as_int32();
    CHECK_GE(axis_val, -1) << "wrong axis: " << axis_val << std::endl;
    CHECK_GE(shape1.size(), shape2.size()) << "A's shape should no less than B's when axis is defined\n";
    CHECK_LE(axis_val, shape1.size() - shape2.size()) << "wrong axis: " << axis_val << std::endl;
    if (axis_val >= 0) {
      int axis_offset = shape1.size() - shape2.size() - axis_val;
      for (int i = 0; i < axis_offset; ++i) {
        // specified axis to align, we push the Expr one to align
        shape2_new.emplace_back(Expr(1));
      }
    }
  }
  int size1 = shape1.size();
  int size2 = shape2_new.size();
  Expr one(1);
  int i;
  for (i = 1; i <= std::min(size1, size2); ++i) {
    const _Var_* var1 = shape1[size1 - i].As<_Var_>();
    const _Var_* var2 = shape2_new[size2 - i].As<_Var_>();
    if (MathEqual(shape1[size1 - i], shape2_new[size2 - i])) {
      common_shape->insert(common_shape->begin(), shape1[size1 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (MathEqual(one, shape1[size1 - i])) {
      CHECK(!MathEqual(one, shape2_new[size2 - i]));
      common_shape->insert(common_shape->begin(), shape2_new[size2 - i]);
      broadcast_flag1->emplace_back(false);
      broadcast_flag2->emplace_back(true);
    } else if (MathEqual(one, shape2_new[size2 - i])) {
      CHECK(!MathEqual(one, shape1[size1 - i]));
      common_shape->insert(common_shape->begin(), shape1[size1 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(false);
    } else if (var1 && var2) {
      Expr max_var = Max::Make(shape1[size1 - i], shape2_new[size2 - i]);
      common_shape->insert(common_shape->begin(), max_var);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (var1) {
      common_shape->insert(common_shape->begin(), shape2_new[size2 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else if (var2) {
      common_shape->insert(common_shape->begin(), shape1[size1 - i]);
      broadcast_flag1->emplace_back(true);
      broadcast_flag2->emplace_back(true);
    } else {
      CHECK(false) << "Incompatible broadcast dims " << shape1[size1 - i] << " and " << shape2_new[size2 - i]
                   << " in: " << shape1 << " and " << shape2_new << std::endl;
    }
  }
  if (size1 != size2) {
    int max_size = std::max(size1, size2);
    auto& shape  = (size1 > size2) ? shape1 : shape2_new;
    auto var_l   = (size1 > size2) ? broadcast_flag1 : broadcast_flag2;
    auto var_s   = (size1 > size2) ? broadcast_flag2 : broadcast_flag1;
    for (; i <= max_size; ++i) {
      common_shape->insert(common_shape->begin(), shape[max_size - i]);
      var_l->emplace_back(true);
      var_s->emplace_back(false);
    }
  }
}

void GetBroadcastIndice(const std::vector<Expr>& indice,
                        std::vector<Expr>* broadcast_indice1,
                        std::vector<Expr>* broadcast_indice2,
                        const std::vector<bool>& broadcast_flags1,
                        const std::vector<bool>& broadcast_flags2) {
  CHECK(broadcast_indice1);
  CHECK(broadcast_indice2);
  if (broadcast_indice1->empty() && broadcast_indice2->empty()) {
    int flag_size = broadcast_flags1.size();
    int i;
    CHECK_GE(indice.size(), flag_size);
    for (i = 0; i < flag_size; i++) {
      if (broadcast_flags1[flag_size - 1 - i]) {
        broadcast_indice1->push_back(indice[i]);
      }
      if (broadcast_flags2[flag_size - 1 - i]) {
        broadcast_indice2->push_back(indice[i]);
      }
    }
  }
}

template <typename FuncOp>
Tensor Broadcast(const FuncOp& op,
                 const Tensor& a,
                 const Tensor& b,
                 const std::string& output_name = "",
                 const Expr& axis               = Expr(-1)) {
  std::vector<Expr> common_shape;
  std::vector<bool> broadcast_flags1;
  std::vector<bool> broadcast_flags2;
  std::vector<Expr> broadcast_indice1;
  std::vector<Expr> broadcast_indice2;

  LOG(INFO) << "a: " << a;
  LOG(INFO) << "b: " << b;
  GetBroadcastShape(a->shape, b->shape, &common_shape, &broadcast_flags1, &broadcast_flags2, axis);
  auto fn = [&](const std::vector<Expr>& indice) {
    GetBroadcastIndice(indice, &broadcast_indice1, &broadcast_indice2, broadcast_flags1, broadcast_flags2);
    return op(a(broadcast_indice1), b(broadcast_indice2));
  };
  Tensor output = Compute(common_shape, fn, output_name);
  return output;
}

#define HLIR_IMP_BC_PE(name__, compute__)                                                             \
  Tensor name__(const Tensor& A, const Tensor& B, const std::string& output_name, const Expr& axis) { \
    auto fn = [&](const Expr& a, const Expr& b) { compute__ };                                        \
    return Broadcast(fn, A, B, output_name, axis);                                                    \
  }                                                                                                   \
  Tensor name__(const Tensor& A, const Expr& B, const std::string& output_name) {                     \
    auto fn = [&](const Expr& a, const Expr& b) { compute__ };                                        \
    return Compute(                                                                                   \
        A->shape, [&](const std::vector<Expr>& indice) { return fn(A(indice), B); }, output_name);    \
  }                                                                                                   \
  Tensor name__(const Expr& A, const Tensor& B, const std::string& output_name) {                     \
    auto fn = [&](const Expr& a, const Expr& b) { compute__ };                                        \
    return Compute(                                                                                   \
        B->shape, [&](const std::vector<Expr>& indice) { return fn(A, B(indice)); }, output_name);    \
  }                                                                                                   \
  Expr name__(const Expr& a, const Expr& b) { compute__ }

HLIR_IMP_BC_PE(Add, return a + b;);
HLIR_IMP_BC_PE(Substract, return a - b;);
HLIR_IMP_BC_PE(Multiply, return a * b;);
HLIR_IMP_BC_PE(Divide, return a / b;);
HLIR_IMP_BC_PE(FloorDivide, return Floor(a / b););
HLIR_IMP_BC_PE(Mod, return a % b;);
HLIR_IMP_BC_PE(FloorMod, return a - Floor(a / b) * b;);
HLIR_IMP_BC_PE(Maximum, return Max::Make(a, b););
HLIR_IMP_BC_PE(Minimum, return Min::Make(a, b););
HLIR_IMP_BC_PE(Power, return Power::Make(a, b););
HLIR_IMP_BC_PE(LeftShift, return a << b;);
HLIR_IMP_BC_PE(RightShift, return a >> b;);
HLIR_IMP_BC_PE(LogicaAnd, return a && b;);
HLIR_IMP_BC_PE(LogicalOr, return a || b;);
HLIR_IMP_BC_PE(LogicalXOr, return a ^ b;);
HLIR_IMP_BC_PE(BitwiseAnd, return a & b;);
HLIR_IMP_BC_PE(BitwiseOr, return a | b;);
HLIR_IMP_BC_PE(BitwiseXor, return a ^ b;);
HLIR_IMP_BC_PE(Greater, return a > b;);
HLIR_IMP_BC_PE(Less, return a < b;);
HLIR_IMP_BC_PE(Equal, return EQ::Make(a, b););
HLIR_IMP_BC_PE(NotEqual, return NE::Make(a, b););
HLIR_IMP_BC_PE(GreaterEqual, return a >= b;);
HLIR_IMP_BC_PE(LessEqual, return a <= b;);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn

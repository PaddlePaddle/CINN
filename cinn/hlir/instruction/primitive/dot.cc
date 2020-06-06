#include "cinn/hlir/instruction/primitive/dot.h"

#include "cinn/common/ir_util.h"
#include "cinn/hlir/instruction/lower_impl.h"
#include "cinn/hlir/instruction/module_lower.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

using cinn::Compute;
using cinn::Expr;
using cinn::Var;

Tensor DotBasicImpl::VecDotVec(const Tensor &a, const Tensor &b, const std::string &name) {
  CHECK(a->SameShapeWith(b)) << "Tensor " << a->name << " and " << b->name << " shape not match for DOT primitive";
  CHECK_EQ(a->shape.size(), 1UL) << "Vector DOT should input one-dimentional tensors";

  Var k(a->shape[0], ctx_->new_var_name("_rv"));
  return cinn::Compute({Expr(1)}, [=]() -> Expr { return cinn::Sum(a(k) * b(k)); }, name, {k});
}

Tensor DotBasicImpl::MatDotMat(const Tensor &a, const Tensor &b, const std::string &name) {
  CHECK_GE(a->shape.size(), 2UL) << "Matrix DOT, the first input tensor should have 2 dimensions";
  CHECK_EQ(b->shape.size(), 2UL) << "Matrix DOT, the second input tensor should have 2 dimensions";
  CHECK(cinn::common::MathEqual(a->shape.back(), b->shape[0]))
      << "1th-input's shape[1] should equal to 2th-input's shape[0], but get " << a->shape[1] << " vs " << b->shape[0];
  Var k(a->shape.back(), ctx_->new_var_name("_rv"));

  Tensor out_tensor;
  switch (a->shape.size()) {
    case 2: {
      auto fn = [=](Var i, Var j) -> Expr { return cinn::Sum(a(i, k) * b(k, j)); };
      std::vector<Expr> shape({a->shape[0], b->shape[1]});
      out_tensor = Compute(shape, fn, name, {k});
    } break;
    case 3: {
      auto fn = [=](Var i0, Var i1, Var j) -> Expr { return cinn::Sum(a(i0, i1, k) * b(k, j)); };
      std::vector<Expr> shape({a->shape[0], a->shape[1], b->shape[1]});
      out_tensor = Compute(shape, fn, name, {k});
    } break;
    case 4: {
      auto fn = [=](Var i0, Var i1, Var i2, Var j) -> Expr { return cinn::Sum(a(i0, i1, i2, k) * b(k, j)); };
      std::vector<Expr> shape({a->shape[0], a->shape[1], a->shape[2], b->shape[1]});
      out_tensor = Compute(shape, fn, name, {k});
    } break;

    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

Tensor DotBasicImpl::MatDotVec(const Tensor &a, const Tensor &b, const std::string &name) {
  CHECK_EQ(a->shape.size(), 2UL);
  CHECK_EQ(b->shape.size(), 1UL);
  CHECK(cinn::common::MathEqual(a->shape[1], b->shape[0]))
      << "shape not match, 1th-input's shape[1] should equal to 2th-input's shape[0], but get " << a->shape[1] << " vs "
      << b->shape[0];

  Var k(a->shape[1], ctx_->new_var_name("_rv"));
  auto fn = [=](Var i) -> Expr { return cinn::Sum(a(i, k) * b(k)); };

  std::vector<Expr> shape({a->shape[0]});
  return Compute(shape, fn, name, {k});
}

Tensor DotBasicImpl::operator()(const Tensor &a, const Tensor &b, const std::string &name) {
  size_t a_dims = a->shape.size();
  size_t b_dims = b->shape.size();

  Tensor res;
  if (a_dims >= 2 && b_dims == 2) {
    res = MatDotMat(a, b, name);
  } else if (a_dims == 2 && b_dims == 1) {
    res = MatDotVec(a, b, name);
  } else if (a_dims == 1 && b_dims == 1) {
    res = VecDotVec(a, b, name);
  } else {
    NOT_IMPLEMENTED
  }

  res->WithBuffer();

  return res;
}

class DotLowerImpl : public LowerImplBase {
 public:
  DotLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(const Instruction *instr, Context *context, Scope *scope, ComputationLower *lower) override {
    CHECK_EQ(instr->operand_count(), 2UL) << "Dot should take two arguments";
    Expr x = scope->Lookup(instr->operand(0));
    Expr y = scope->Lookup(instr->operand(1));
    CHECK(x.defined());
    CHECK(y.defined());

    auto out = DotBasicImpl(context)(x.as_tensor_ref(), y.as_tensor_ref(), context->new_var_name("dot_out"));
    scope->Insert(instr, out);
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

REGISTER_INSTRUCTION_LOWER(base, Dot, DotLowerImpl)

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
      CINN_NOT_IMPLEMENTED
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
    CINN_NOT_IMPLEMENTED
  }

  return res;
}

class DotLowerImpl : public LowerImplBase {
 public:
  explicit DotLowerImpl(InstrCode code) : LowerImplBase(code) {}

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

class DotCblasLowerImpl : public LowerImplBase {
 public:
  explicit DotCblasLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(const Instruction *instr, Context *context, Scope *scope, ComputationLower *lower) override {
    CHECK_EQ(instr->operand_count(), 2UL) << "Dot should take two arguments";
    Expr x = scope->Lookup(instr->operand(0));
    Expr y = scope->Lookup(instr->operand(1));
    CHECK(x.defined());
    CHECK(y.defined());

    auto *xt = x.as_tensor();
    auto *yt = y.as_tensor();
    CHECK(xt);
    CHECK(yt);
    CHECK(!xt->inlined()) << "x should bind to a buffer";
    CHECK(!yt->inlined()) << "y should bind to a buffer";

    CHECK_GE(xt->shape.size(), 2UL);
    CHECK_EQ(yt->shape.size(), 2UL);
    CHECK(cinn::common::MathEqual(xt->shape.back(), yt->shape[0]));

    Expr K = yt->shape[0];
    Expr N = yt->shape[1];
    Expr M = xt->shape[0];
    for (int i = 1; i < xt->shape.size() - 1; i++) M = M * xt->shape[i];

    auto call = Compute(
        {Expr(1)},
        [=]() -> Expr {
          return lang::CallExtern("cinn_cpu_mkl_gemm_fp32",
                                  {
                                      common::make_one<float>(),   // alpha
                                      M,                           // M
                                      N,                           // N
                                      K,                           // K
                                      common::make_bool(false),    // ta
                                      common::make_bool(false),    // tb
                                      M,                           // lda
                                      K,                           // ldb
                                      M,                           // ldc
                                      common::make_zero<float>(),  // beta
                                      x,                           // A
                                      y,                           // B
                                  });
        },
        context->new_ssa_id("cinn_cpu_mkl_gemm_fp32_call"));
    auto out = call->TupleGet(0);

    scope->Insert(instr, out);
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

REGISTER_INSTRUCTION_LOWER(base, Dot, cinn::hlir::instruction::primitive::DotLowerImpl)
REGISTER_INSTRUCTION_LOWER(cblas, Dot, cinn::hlir::instruction::primitive::DotLowerImpl)

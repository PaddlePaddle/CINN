#include "cinn/hlir/instruction/primitive/binary.h"

#include "cinn/hlir/instruction/context.h"
#include "cinn/hlir/instruction/instruction.h"
#include "cinn/hlir/instruction/lower_impl.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {
using cinn::ir::_Tensor_;
using cinn::ir::Expr;
using cinn::ir::Tensor;
using cinn::ir::Var;
using cinn::lang::Compute;

cinn::ir::Tensor BinaryBasicImpl::operator()(const cinn::ir::Tensor& a,
                                             const cinn::ir::Tensor& b,
                                             const std::string& name) {
  CHECK(a.defined());
  CHECK(b.defined());

  int ndims = a->shape.size();

  std::vector<Expr> shape;
  Tensor out_tensor;
  switch (b->shape.size()) {
    case 1:
      out_tensor = RunWithArgb1Dim(a, b);
      break;
    case 2:
      out_tensor = RunWithArgb2Dim(a, b);
      break;
    case 3:
      out_tensor = RunWithArgb3Dim(a, b);
      break;
    case 4:
      out_tensor = RunWithArgb4Dim(a, b);
      break;
    case 5:
      out_tensor = RunWithArgb5Dim(a, b);
      break;
    default:
      NOT_IMPLEMENTED
  }

  if (!inlined_) out_tensor->WithBuffer();
  return out_tensor;
}

cinn::ir::Tensor BinaryBasicImpl::RunWithArgb1Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 1UL);

  Tensor out_tensor;
  switch (a->shape.size()) {
    case 1:
      out_tensor = Compute(a->shape, [a, b, this](Var i) -> Expr { return opr_(a(i), b(i)); });
      break;
    case 2:
      out_tensor = Compute(a->shape, [a, b, this](Var i, Var j) -> Expr { return opr_(a(i, j), b(j)); });
      break;
    case 3:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var j) -> Expr { return opr_(a(i0, i1, j), b(j)); });
      break;
    case 4:
      out_tensor = Compute(
          a->shape, [a, b, this](Var i0, Var i1, Var i2, Var j) -> Expr { return opr_(a(i0, i1, i2, j), b(j)); });
      break;
    case 5:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_(a(i0, i1, i2, i3, j), b(j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

cinn::ir::Tensor BinaryBasicImpl::RunWithArgb2Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 2UL);
  auto opr_copied = opr_;
  ir::Tensor out_tensor;
  switch (a->shape.size()) {
    case 2:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i, Var j) -> Expr { return opr_copied(a(i, j), b(i, j)); });
      break;
    case 3:
      out_tensor = Compute(
          a->shape, [a, b, opr_copied](Var i0, Var i1, Var j) -> Expr { return opr_copied(a(i0, i1, j), b(i1, j)); });
      break;
    case 4:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i0, Var i1, Var i2, Var j) -> Expr {
        return opr_copied(a(i0, i1, i2, j), b(i2, j));
      });
      break;
    case 5:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_copied(a(i0, i1, i2, i3, j), b(i3, j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

cinn::ir::Tensor BinaryBasicImpl::RunWithArgb3Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 3UL);
  Tensor out_tensor;
  auto opr_copied = opr_;
  switch (a->shape.size()) {
    case 3:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i0, Var i1, Var j) -> Expr {
        return opr_copied(a(i0, i1, j), b(i0, i1, j));
      });
      break;
    case 4:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i0, Var i1, Var i2, Var j) -> Expr {
        return opr_copied(a(i0, i1, i2, j), b(i1, i2, j));
      });
      break;
    case 5:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_copied(a(i0, i1, i2, i3, j), b(i2, i3, j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

cinn::ir::Tensor BinaryBasicImpl::RunWithArgb4Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 4UL);
  Tensor out_tensor;
  auto opr_copied = opr_;
  switch (a->shape.size()) {
    case 4:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i0, Var i1, Var i2, Var j) -> Expr {
        return opr_copied(a(i0, i1, i2, j), b(i0, i1, i2, j));
      });
      break;
    case 5:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_copied(a(i0, i1, i2, i3, j), b(i1, i2, i3, j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

cinn::ir::Tensor BinaryBasicImpl::RunWithArgb5Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 5UL);
  Tensor out_tensor;
  auto opr_copied = opr_;
  switch (a->shape.size()) {
    case 5:
      out_tensor = Compute(a->shape, [a, b, opr_copied](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_copied(a(i0, i1, i2, i3, j), b(i0, i1, i2, i3, j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

class BinaryLowerImpl : public LowerImplBase {
 public:
  explicit BinaryLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(const Instruction* instr, Context* context, Scope* scope, ComputationLower* lower) override {
    auto a = scope->Lookup(instr->operand(0));
    auto b = scope->Lookup(instr->operand(1));
    CHECK(a.defined());
    CHECK(b.defined());
    auto at = a.as_tensor_ref();
    auto bt = b.as_tensor_ref();
    Expr out;
    switch (instr->instr_code()) {
      case InstrCode::Add:
        out = BinaryBasicImpl(
            context, [](Expr a, Expr b) { return a + b; }, instr->inlined())(at, bt, context->new_ssa_id("add"));
        break;
      case InstrCode::Sub:
        out = BinaryBasicImpl(
            context, [](Expr a, Expr b) { return a - b; }, instr->inlined())(at, bt, context->new_ssa_id("add"));
        break;
      case InstrCode::Mul:
        out = BinaryBasicImpl(
            context, [](Expr a, Expr b) { return a * b; }, instr->inlined())(at, bt, context->new_ssa_id("add"));
        break;
      case InstrCode::Div:
        out = BinaryBasicImpl(
            context, [](Expr a, Expr b) { return a / b; }, instr->inlined())(at, bt, context->new_ssa_id("add"));
        break;
      default:
        NOT_IMPLEMENTED
    }

    scope->Insert(instr, out);
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

REGISTER_INSTRUCTION_LOWER(base, Add, cinn::hlir::instruction::primitive::BinaryLowerImpl);
REGISTER_INSTRUCTION_LOWER(base, Sub, cinn::hlir::instruction::primitive::BinaryLowerImpl);
REGISTER_INSTRUCTION_LOWER(base, Mul, cinn::hlir::instruction::primitive::BinaryLowerImpl);
REGISTER_INSTRUCTION_LOWER(base, Div, cinn::hlir::instruction::primitive::BinaryLowerImpl);

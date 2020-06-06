#include "cinn/hlir/instruction/primitive/elementwise.h"

#include "cinn/hlir/instruction/lower_impl.h"
#include "cinn/hlir/instruction/module_lower.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

struct ElementwiseLower {
  using opr_t = std::function<Expr(Expr)>;

  explicit ElementwiseLower(Type type, opr_t&& func, bool inlined = true, ir::Buffer buffer = ir::Buffer())
      : type_(type), opr_(std::move(func)), inlined_(inlined), buffer_(buffer) {}

  Expr const_expr(double v) const { return make_const(type_, v); }

  ir::Tensor operator()(const ir::Tensor& a, const std::string& name) {
    auto t = Compute(
        a->shape, [=](const std::vector<Expr>& args) -> Expr { return opr_(a(args)); }, name);
    if (inlined_) {
      if (buffer_.defined()) {
        t->Bind(buffer_);
      } else {
        t->WithBuffer();
      }
    }
    return t;
  }

 private:
  Type type_;
  opr_t opr_;
  bool inlined_;
  ir::Buffer buffer_;
};

ir::Tensor Abs(const ir::Tensor& a, const std::string& name) {
  ElementwiseLower lower(a->type(), [=](Expr x) {
    return ir::Select::Make(x > make_const(a->type(), 0), x, x * make_const(a->type(), -1));
  });
  return lower(a, name);
}

ir::Tensor Ceil(const ir::Tensor& a, const std::string& name) {
  return Compute(
      a->shape,
      [a](const std::vector<Expr>& indice) -> Expr { return ir::Activate::Make(ir::Activate::Kind::kCeil, a(indice)); },
      name);
}

ir::Tensor Floor(const ir::Tensor& a, const std::string& name) {
  return Compute(
      a->shape,
      [a](const std::vector<Expr>& indice) -> Expr {
        return ir::Activate::Make(ir::Activate::Kind::kFloor, a(indice));
      },
      name);
}

ir::Tensor Sign(const ir::Tensor& a, const std::string& name) {
  auto zero    = make_const(a->type(), 0);
  auto one     = make_const(a->type(), 1);
  auto neg_one = make_const(a->type(), -1);
  ElementwiseLower lower(a->type(), [=](Expr x) { return ir::Select::Make(x > zero, one, neg_one); });
  return lower(a, name);
}

ir::Tensor Tanh(const ir::Tensor& a, const std::string& name) {
  return Compute(
      a->shape,
      [a](const std::vector<Expr>& indice) -> Expr { return ir::Activate::Make(ir::Activate::Kind::kTanh, a(indice)); },
      name);
}

ir::Tensor Sigmoid(const ir::Tensor& a, const std::string& name) {
  return Compute(
      a->shape,
      [a](const std::vector<Expr>& indice) -> Expr {
        return ir::Activate::Make(ir::Activate::Kind::kSigmoid, a(indice));
      },
      name);
}

ir::Tensor Exp(const ir::Tensor& a, const std::string& name) {
  return Compute(
      a->shape,
      [a](const std::vector<Expr>& indice) -> Expr { return ir::Activate::Make(ir::Activate::Kind::kExp, a(indice)); },
      name);
}

struct ElementwiseLowerImpl : public LowerImplBase {
  ElementwiseLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(const Instruction* instr, Context* context, Scope* scope, ComputationLower* lower) override {
    CHECK_EQ(instr->operand_count(), 1UL) << "Elementwise instruction should take only one argument";
    Expr x = scope->Lookup(instr->operand(0));
    CHECK(x.defined()) << "Tensor not found for instruction: " << instr->operand(0)->to_debug_string();
    switch (code()) {
#define __(code__)                                                             \
  case InstrCode::code__: {                                                    \
    auto out = code__(x.as_tensor_ref(), context->new_ssa_id(#code__ "_out")); \
    if (!instr->inlined()) {                                                   \
      out->WithBuffer();                                                       \
    }                                                                          \
    scope->Insert(instr, out);                                                 \
  } break;

      __(Tanh)
      __(Ceil)
      __(Abs)
      __(Sign)
      __(Exp)

      default:
        LOG(FATAL) << "ElementwiseLowerImpl not support op " << code();

#undef __
    }
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

REGISTER_INSTRUCTION_LOWER(base, Tanh, ElementwiseLowerImpl)
REGISTER_INSTRUCTION_LOWER(base, Ceil, ElementwiseLowerImpl)
REGISTER_INSTRUCTION_LOWER(base, Abs, ElementwiseLowerImpl)
REGISTER_INSTRUCTION_LOWER(base, Exp, ElementwiseLowerImpl)

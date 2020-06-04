#include "cinn/hlir/instruction/primitive/elementwise.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

struct ElementwiseLower {
  using opr_t = std::function<Expr(Expr)>;

  explicit ElementwiseLower(Type type, opr_t&& func, bool inlined = true, ir::Buffer buffer = ir::Buffer())
      : type_(type), opr_(std::move(func)), inlined_(inlined), buffer_(buffer) {}

  Expr const_expr(double v) const { return make_const(type_, v); }

  ir::Tensor operator()(const ir::Tensor& a) {
    auto t = Compute(a->shape, [=](const std::vector<Expr>& args) -> Expr { return opr_(a(args)); });
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
  return lower(a);
}

ir::Tensor Ceil(const ir::Tensor& a, const std::string& name) {
  return Compute(a->shape, [a](const std::vector<Expr>& indice) -> Expr {
    return ir::Activate::Make(ir::Activate::Kind::kCeil, a(indice));
  });
}

ir::Tensor Floor(const ir::Tensor& a, const std::string& name) {
  return Compute(a->shape, [a](const std::vector<Expr>& indice) -> Expr {
    return ir::Activate::Make(ir::Activate::Kind::kFloor, a(indice));
  });
}

ir::Tensor Sign(const ir::Tensor& a, const std::string& name) {
  auto zero    = make_const(a->type(), 0);
  auto one     = make_const(a->type(), 1);
  auto neg_one = make_const(a->type(), -1);
  ElementwiseLower lower(a->type(), [=](Expr x) { return ir::Select::Make(x > zero, one, neg_one); });
  return lower(a);
}

ir::Tensor Tanh(const ir::Tensor& a, const std::string& name) {
  return Compute(a->shape, [a](const std::vector<Expr>& indice) -> Expr {
    return ir::Activate::Make(ir::Activate::Kind::kTanh, a(indice));
  });
}

ir::Tensor Exp(const ir::Tensor& a, const std::string& name) {
  return Compute(a->shape, [a](const std::vector<Expr>& indice) -> Expr {
    return ir::Activate::Make(ir::Activate::Kind::kExp, a(indice));
  });
}

ir::Tensor Sigmoid(const ir::Tensor& a, const std::string& name) {
  // 1 / exp(-x)
  return Compute(a->shape, [a](const std::vector<Expr>& indice) -> Expr {
    return make_const(a->type(), 1.f) /
           ir::Activate::Make(ir::Activate::Kind::kExp, make_const(a->type(), -1.f) * a(indice));
  });
}

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

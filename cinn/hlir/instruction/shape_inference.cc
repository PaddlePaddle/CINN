#include "cinn/hlir/instruction/shape_inference.h"

#include "cinn/hlir/instruction/instruction.h"

namespace cinn {
namespace hlir {
namespace instruction {

Shape BinaryInferenceShape(Instruction *a, Instruction *b) {
  CHECK(a->shape() == b->shape() || b->shape().is_identity());
  return a->shape();
}

struct DotInferenceShapeVisitor {
  DotInferenceShapeVisitor(Shape *shape) : shape_(shape) {}

  void operator()(int v) { shape_->AddDim(v); }
  void operator()(const ir::Var &v) { shape_->AddDim(v); }

 private:
  Shape *shape_{};
};

Shape DotInferenceShape(Instruction *a, Instruction *b) {
  auto a_shape = a->shape();
  auto b_shape = b->shape();

  Shape res;
  DotInferenceShapeVisitor visitor(&res);

  for (int i = 0; i < a_shape.num_dims() - 1; i++) {
    std::visit(visitor, a_shape[i]);
  }

  for (int i = 1; i < b_shape.num_dims(); i++) {
    std::visit(visitor, b_shape[i]);
  }
  return res;
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
#pragma once

#include <cinn/ir/node.h>
namespace cinn {
namespace lang {
using ir::Expr;
using ir::IrNodeRef;
using ir::Type;

class _Tensor_;

/**
 * @brief Tensor representing a possible input, or intermediate computation result.
 */
class Tensor : public ir::IrNode {
 public:
  Tensor() = default;
  explicit Tensor(IrNode* n) : n_(n) {}

 private:
  ir::IrNodeRef n_;
};

class _Tensor_ : public ir::IrNodeRef {
 public:
  std::vector<Expr> shape;
  Type dtype;
};

}  // namespace lang
}  // namespace cinn

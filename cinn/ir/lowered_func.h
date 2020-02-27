#pragma once
#include "cinn/ir/buffer.h"
#include "cinn/ir/node.h"

namespace cinn {
namespace ir {

class _LoweredFunc_;

/**
 * A struct representing an argument to a lowered function. Used for specifying the function signature of generated
 * code.
 */
struct Argument {
  //! The name of the argument.
  std::string name;

  enum class Kind { kScalar = 0, kBuffer } kind{Kind::kScalar};

  //! Number of the dimensions of buffer.
  uint32_t ndims{0};

  //! The type of the buffer or scalar.
  Type type;

  bool is_buffer() const { return kind == Kind::kBuffer; }
  bool is_scalar() const { return kind == Kind::kScalar; }

  Argument() {}
  Argument(const std::string& name, Kind kind, const Type& type, int ndims)
      : name(name), kind(kind), type(type), ndims(ndims) {}

  explicit Argument(const ir::Buffer& buffer) : name(buffer->name), type(buffer->type()), ndims(buffer->shape.size()) {}
};

//! Wrapper for _LoweredFunc_
class LoweredFunc : public IrNodeRef {
 public:
  LoweredFunc() = default;
  explicit LoweredFunc(IrNode* n) : IrNodeRef(n) {}

  operator Expr() const { return Expr(ptr()); }

  const _LoweredFunc_* operator->() const;
  _LoweredFunc_* operator->();
};

/**
 * Definition of a lowered function. Note that, it should be functional.
 */
struct _LoweredFunc_ : ExprNode<_LoweredFunc_> {
  //! The name of this function.
  std::string name;

  //! The Arguments used in the body of the function.
  std::vector<Argument> args;

  //! Body of this function.
  Expr body;

  static LoweredFunc Make(const std::string& name, const std::vector<Argument>& args, const Expr& body);

  static LoweredFunc Make(const std::string& name, const std::vector<Argument>& args, const std::vector<Expr>& body);

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_LoweredFunc_;
};

}  // namespace ir
}  // namespace cinn

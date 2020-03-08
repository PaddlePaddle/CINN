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
  enum class Kind { kScalar = 0, kBuffer };
  //! Input or output.
  enum class IO { kInput = 0, kOutput = 1 };

  Kind kind{Kind::kScalar};
  IO io{IO::kInput};

  //! Number of the dimensions of buffer.
  uint32_t ndims{0};

  //! Set the buffer field.
  void set_buffer(const ir::Buffer& x);

  //! Set the scalar field.
  void set_scalar(const ir::Var& x);

  bool is_buffer() const { return kind == Kind::kBuffer; }
  bool is_scalar() const { return kind == Kind::kScalar; }

  bool is_input() const { return io == IO::kInput; }
  bool is_output() const { return io == IO::kOutput; }

  Argument() = default;
  Argument(const ir::Buffer& buffer, IO io = IO::kInput) {
    set_buffer(buffer);
    this->io    = io;
    this->ndims = buffer->shape.size();
  }

  Argument(const ir::Var& var, IO io = IO::kInput) {
    set_scalar(var);
    this->io    = io;
    this->ndims = 1;
  }

  ir::Buffer buffer_arg() const;
  ir::Var scalar_arg() const;

  //! The type of the buffer or scalar.
  Type type() const {
    if (is_scalar())
      return scalar_arg()->type();
    else if (is_buffer())
      return buffer_arg()->type();
    else
      NOT_IMPLEMENTED
  }

  std::string name() const {
    if (is_buffer())
      return buffer_arg()->name;
    else if (is_scalar())
      return scalar_arg()->name;
    else
      NOT_IMPLEMENTED
    return "";
  }

 private:
  //! The buffer field.
  ir::Buffer buffer_arg_;
  //! The scalar field.
  ir::Var scalar_arg_;
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

  std::vector<Expr> alloc_output_buffer_exprs;
  std::vector<Expr> alloc_tmp_buffer_exprs;
  //! something like: float* A_data = (float*)(A->host_memory);
  std::vector<Expr> buffer_data_cast_exprs;

  static LoweredFunc Make(const std::string& name, const std::vector<Argument>& args, const Expr& body);

  static LoweredFunc Make(const std::string& name, const std::vector<Argument>& args, const std::vector<Expr>& body);

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_LoweredFunc_;

 private:
  void CheckValid() const;
  //! Insert the allocation buffer for outputs.
  void AllocBufferForOutputs();
  //! Insert the allocation expr for temporary variables.
  void AllocTempBuffer();
  void PrepareBufferCastExprs();
  //! Get all the Buffers the function body references.
  //! NOTE it will return the buffers with duplicates removed(by comparing their name).
  std::vector<Tensor> CollectAllTensorReference() const;
};

}  // namespace ir
}  // namespace cinn

#pragma once
#include <map>
#include <string>
#include <vector>

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
  //! Input or output.
  enum class IO { kInput = 0, kOutput = 1 };

  IO io{IO::kInput};

  Argument() = default;
  explicit Argument(const ir::Buffer& buffer, IO io = IO::kInput);
  explicit Argument(const ir::Var& var, IO io = IO::kInput);

  //! Set the buffer argument, all the buffer information are stored in ir::Buffer.
  void set_buffer(const ir::Buffer& x);

  //! Set the var argument.
  void set_var(const ir::Var& x);

  bool is_input() const { return io == IO::kInput; }
  bool is_output() const { return io == IO::kOutput; }

  bool is_var() const { return var_arg_.defined(); }
  bool is_buffer() const { return buffer_arg_.defined(); }
  bool defined() const { return is_var() || is_buffer(); }

  ir::Buffer buffer_arg() const;
  ir::Var var_arg() const;

  //! The type of the buffer or scalar.
  Type type() const;

  std::string name() const;

  std::string human_readable() const;

 private:
  //! The buffer field.
  ir::Buffer buffer_arg_;
  //! The scalar field.
  ir::Var var_arg_;
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

  std::vector<int> gpu_grid_dims;
  std::vector<int> gpu_block_dims;

  /**
   * The output buffer will be resized to the size required, we leave all the expression here.
   * The allocation and deallocation expressions will insert into the head and tail of the function's body. It supports
   * lazy allocation/deallocation if the corresponding intristic methods support.
   *
   * Currently, we assume that all the input and output buffers should locate in heap, no other memory type is allowed.
   */
  // @{
  std::vector<Expr> alloc_output_buffer_exprs;
  std::vector<Expr> dealloc_output_buffer_exprs;
  // @}

  std::vector<Expr> alloc_tmp_buffer_exprs;
  //! something like: float* A_data = (float*)(A->host_memory);
  std::vector<Expr> buffer_data_cast_exprs;

  std::vector<Expr> argument_prepare_exprs;

  static LoweredFunc Make(const std::string& name, const std::vector<Argument>& args, const Expr& body);

  static LoweredFunc Make(const std::string& name, const std::vector<Argument>& args, const std::vector<Expr>& body);

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_LoweredFunc_;

 private:
  void CheckValid() const;
  //! Prepare the expressions for `alloc_output_buffer_exprs`.
  void PrepareAllocOutputBufferExprs();
  //! Prepare the expressions for `dealloc_output_buffer_exprs`.
  void PrepareDeallocOutputBufferExprs();
  //! Insert the allocation expr for temporary variables.
  void AllocTempBuffer();
  void PrepareBufferCastExprs();
  void PrepareArgumentExprs();
  //! Get all the Buffers the function body references.
  //! NOTE it will return the buffers with duplicates removed(by comparing their name).
  std::vector<Tensor> CollectAllTensorReference() const;
};

}  // namespace ir
}  // namespace cinn

#pragma once

#include <isl/cpp.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/common/graph_utils.h"
#include "cinn/ir/function_base.h"
#include "cinn/ir/ir.h"

namespace cinn {

namespace poly {

struct Stage;

}  // namespace poly

namespace ir {

namespace detail {
constexpr bool LE(int a, int b) { return a <= b; }
constexpr bool GE(int a, int b) { return a >= b; }

//! Expand milti-dim indices to 1-dim index.
Expr ExpandTo1DIndice(const std::vector<int>& shape, const std::vector<Expr>& indices);
Expr ExpandTo1DIndice(const std::vector<Expr>& shape, const std::vector<Expr>& indices);

}  // namespace detail

class _Tensor_;

/**
 * @brief Tensor representing a possible input, or intermediate computation result.
 * Tensor is the most general type in CINN, it holds the computation, or placeholder. A tensor can `store_at` a Buffer,
 * or just has a expression and easy to inline expanded in the consumer's computation.
 */
class Tensor : public ir::IrNodeRef {
 public:
  Tensor() = default;
  explicit Tensor(ir::IrNode* n) : IrNodeRef(n) {}
  /**
   * Constructor.
   * @param shape The shape of the tensor.
   * @param axis The iterators to use.
   * @param dtype The data type of the tensor.
   * @param expr The expression to compute this tensor.
   */
  Tensor(const std::vector<Expr>& shape,
         const std::vector<Var>& axis,
         Type dtype,
         Expr expr,
         const std::string& name = "");

  //! Get number of dimensions.
  inline size_t ndims() const;

  /**
   * Take elements from the tensor.
   * This take one or multiple expressions as indices.
   *
   * usage:
   *
   * Tensor A;
   * A(i,j) get the [i][j] element.
   */
  // @{
  Expr operator()(const Expr& a) const { return operator()({a}); }
  template <typename... Args>
  inline typename std::enable_if<detail::GE(sizeof...(Args), 2), Expr>::type operator()(Args... args) const {
    return operator()({std::forward<Args>(args)...});
  }
  // @}

  /**
   * Take elements from the tensor.
   * @param indices  The indices.
   * @return The result expression representing a tensor read.
   */
  Expr operator()(const std::vector<Expr>& indices) const;

  inline const _Tensor_* operator->() const { return As<_Tensor_>(); }
  inline _Tensor_* operator->() { return As<_Tensor_>(); }

  inline operator Expr() const { return Expr(get()); }
};

/**
 * _Tensor_ holds the content of a Tensor.
 */
class _Tensor_ : public ExprNode<_Tensor_> {
 public:
  //! Shape of this tensor.
  std::vector<Expr> shape;
  //! Tensor axis.
  std::vector<Var> axis;
  //! The operation that generates Tensor.
  FunctionRef operaion;
  //! Name of this tensor.
  std::string name;
  //! Polyhedral element for analysis and schedule.
  poly::Stage* stage{};
  //! The binded buffer, for each tensor if it is not inline.
  Var buffer_var;

  //! Generate a tensor from a computation.
  static Tensor Make(const std::string& name,
                     const std::string& tag,
                     const std::vector<Expr>& shape,
                     const std::vector<Var>& axis,
                     Type dtype,
                     const std::map<std::string, IrNodeRef>& attrs,
                     const std::vector<Expr>& body = {});

  //! Generate a tensor from a function.
  static Tensor Make(const std::string& name, const std::vector<Expr>& shape, FunctionRef fn);

  //! Tell the operation type.
  // @{
  bool is_compute_node() const;
  bool is_placeholder_node() const;
  const char* operation_type() const;
  // @}

  //! The expression generate this tensor, will be empty if it is a PlaceHolder.
  Expr body() const;
  //! Get the expression with `store(tensor)` inserted into the body.
  Expr tensor_store_expanded_body() const;

  Expr inline_expanded(const std::vector<Expr>& indices);

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_Tensor_;

  _Tensor_() : ExprNode<_Tensor_>(Float(32)), stage(nullptr) {}

  ~_Tensor_();

 private:
  //! Create the polyhedral element for analysis.
  //! It is based on the shape.
  void InitStage();

  //! Initialize the axis field after the shape field is assigned.
  void InitAxis();

  //! Bind the tensor to a buffer by default.
  //! NOTE it should called by all the Make.
  void SetDefaultBindedBuffer() { buffer_var = ir::_Var_::Make(name, type()).As<_Var_>(); }

  isl::set GenerateIslDomain();
};

class _Operation_;
class Operation : public FunctionRef {
 public:
  Operation() = default;
  explicit Operation(IrNode* n) : FunctionRef(n) {}

  inline const _Operation_* operator->() const;

  //! Get the i-th output of the operation.
  // Tensor output(size_t i) const;

  std::string name;
};

class _Operation_ : public ir::FunctionBase {
 public:
  //! Optional name of the operation.
  std::string name;
  //! Optional tag of the operation.
  std::string tag;
  //! Additional attributes of the operation.
  std::map<std::string, IrNodeRef> attrs;

  void Accept(IRVisitor* v) const override {}
  const std::string& func_name() const final { return name; }
  //! The function type.
  virtual const char* func_type() const = 0;
};

}  // namespace ir
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::ir::Tensor> {
  inline size_t operator()(const cinn::ir::Tensor& x) {
    // We treat the tensor's name as the unique identifier.
    return std::hash<std::string>()(x->name);
  }
};

}  // namespace std

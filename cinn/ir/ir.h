/**
 * This file contains all the internal representations used in CINN project.
 */
#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cinn/common/shared.h"
#include "cinn/common/type.h"
#include "cinn/ir/function_base.h"
#include "cinn/ir/node.h"

namespace cinn {

namespace poly {
class Element;
}  // namespace poly

namespace ir {

using common::Object;
using common::Shared;

/**
 * Cast a node to another type, can't change the width.
 */
struct Cast : public UnaryOpNode<Cast> {
  Cast(Type t, Expr v) : UnaryOpNode<Cast>(t, v) {}

  static Expr Make(Type t, Expr v);

  void Accept(IRVisitor* v) const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Cast;
};

/**
 * The sum of two expressions.
 */
struct Add : public BinaryOpNode<Add> {
  Add(Expr a, Expr b) : BinaryOpNode<Add>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);

  static const IrNodeTy _node_type_ = IrNodeTy::Add;
};

/**
 * The difference of two expressions.
 */
struct Sub : public BinaryOpNode<Sub> {
  Sub(Expr a, Expr b) : BinaryOpNode<Sub>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);

  static const IrNodeTy _node_type_ = IrNodeTy::Sub;
};

/**
 * The product of two expressions.
 */
struct Mul : public BinaryOpNode<Mul> {
  Mul(Expr a, Expr b) : BinaryOpNode<Mul>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::Mul;
};

/**
 * The ratio of two expressions.
 */
struct Div : public BinaryOpNode<Div> {
  Div(Expr a, Expr b) : BinaryOpNode<Div>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::Div;
};

/**
 * The mod of two expressions.
 */
struct Mod : public BinaryOpNode<Mod> {
  Mod(Expr a, Expr b) : BinaryOpNode<Mod>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::Mod;
};

/**
 * The lesser of two expressions.
 */
struct Min : public BinaryOpNode<Min> {
  Min(Expr a, Expr b) : BinaryOpNode<Min>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::Min;
};

/**
 * The larger of two expressions.
 */
struct Max : public BinaryOpNode<Max> {
  Max(Expr a, Expr b) : BinaryOpNode<Max>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);

  static const IrNodeTy _node_type_ = IrNodeTy::Max;
};

/**
 * Tell whether the first expression equals to the second expression.
 */
struct EQ : public BinaryOpNode<EQ> {
  EQ(Expr a, Expr b) : BinaryOpNode<EQ>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::EQ;
};

/**
 * Tell whether the first expression not equals to the second expression.
 */
struct NE : public BinaryOpNode<NE> {
  NE(Expr a, Expr b) : BinaryOpNode<NE>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::NE;
};

/**
 * Tell whether the first expression is lower than the second expression.
 */
struct LT : public BinaryOpNode<LT> {
  LT(Expr a, Expr b) : BinaryOpNode<LT>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::LT;
};

/**
 * Tell whether the first expression is no larger than the second expression.
 */
struct LE : public BinaryOpNode<LE> {
  LE(Expr a, Expr b) : BinaryOpNode<LE>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::LE;
};

/**
 * Tell whether the first expression is larger than the second expression.
 */
struct GT : public BinaryOpNode<GT> {
  GT(Expr a, Expr b) : BinaryOpNode<GT>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::GT;
};

/**
 * Tell whether the first expression is not less than the second expression.
 */
struct GE : public BinaryOpNode<GE> {
  GE(Expr a, Expr b) : BinaryOpNode<GE>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::GE;
};

/**
 * Logical and.
 */
struct And : public BinaryOpNode<And> {
  And(Expr a, Expr b) : BinaryOpNode<And>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  static const IrNodeTy _node_type_ = IrNodeTy::And;
};

/**
 * -x
 */
struct Minus : public UnaryOpNode<Minus> {
  Minus(Expr x) : UnaryOpNode<Minus>(x.type(), x) {}

  static Expr Make(Expr a);
  static const IrNodeTy _node_type_ = IrNodeTy::Minus;
};

/**
 * Logical or.
 */
struct Or : public BinaryOpNode<Or> {
  Or(Expr a, Expr b) : BinaryOpNode<Or>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);

  static const IrNodeTy _node_type_ = IrNodeTy::Or;
};

/**
 * Logical not.
 */
struct Not : public UnaryOpNode<Not> {
  Not(Expr v) : UnaryOpNode<Not>(v.type(), v) {}

  static Expr Make(Expr v);

  static const IrNodeTy _node_type_ = IrNodeTy::Not;
};

struct Call : public ExprNode<Call> {
  Call(Type t) : ExprNode<Call>(t) {}

  enum CallType : int {
    //! Extern "C" function.
    Extern = 0,
    //! Halide-style call.
    Halide,
    //! Intrinsic functions.
    Intrinsic,
    //! Generated from ISL Ast.
    ISL,
  };

  //! The name of the function/intrinsic.
  std::string name;
  //! The arguments.
  std::vector<Expr> args;
  //! Type of calls.
  CallType call_type;
  //! The function to be called.
  FunctionRef func;
  //! The output value index if func's value is a tuple.
  int value_index{};

  static Expr Make(Type type,
                   const std::string& name,
                   const std::vector<Expr>& args,
                   CallType call_type,
                   FunctionRef func = FunctionRef(),
                   int value_index  = 0);

  static const IrNodeTy _node_type_ = IrNodeTy::Call;
};

/**
 * Variable used as iterator value or bound definition.
 */
struct _Var_ : public ExprNode<_Var_> {
  std::string name;

  _Var_(const std::string& name, Type type) : ExprNode<_Var_>(type), name(name) {}

  static Expr Make(const std::string& name, const Type& type);

  static const IrNodeTy _node_type_ = IrNodeTy::_Var_;
};

//! A named variable.
struct Var : public IrNodeRef {
  Var() = default;
  explicit Var(IrNode* n) : IrNodeRef(n) {}
  explicit Var(const std::string& name_hint, Type t = type_of<int>()) : Var(_Var_::Make(name_hint, t).ptr()) {}

  operator Expr() { return Expr(get()); }

  bool operator==(const Var& o) const;
  bool operator!=(const Var& o) const;

  const _Var_* operator->() const { return get(); }
  _Var_* operator->() { return get(); }
  const _Var_* get() const { return static_cast<const _Var_*>(ptr()); }
  _Var_* get() { return static_cast<_Var_*>(ptr()); }
};

/**
 * Evaluates `true_value` and `false_value` then selects between them based on `condition`.
 */
struct Select : public ExprNode<Select> {
  Expr condition;
  Expr true_value;
  Expr false_value;

  Select(Expr condition, Expr true_value, Expr false_value)
      : ExprNode<Select>(true_value.type()), condition(condition), true_value(true_value), false_value(false_value) {
    CHECK_EQ(true_value.type(), false_value.type());
  }

  static Expr Make(Expr condition, Expr true_value, Expr false_value) {
    auto node = new Select(condition, true_value, false_value);
    return Expr(node);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::Select;
};

/**
 * Load the value from a buffer (as an array).
 */
struct Load : public ExprNode<Load> {
  Var buffer_var;  // should be a Variable.
  Expr index;

  Load(Var buffer, Expr index) : ExprNode<Load>(buffer->type().ElementOf()), buffer_var(buffer), index(index) {}

  static Expr Make(Var buffer, Expr index) {
    auto node = new Load(buffer, index);
    return Expr(node);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::Load;
};

/**
 * Store a `value` to the buffer at a given `index`.
 */
struct Store : public ExprNode<Store> {
  Var buffer_var;
  Expr value, index;

  Store() : ExprNode(Type()) {}

  static Expr Make(Var buffer_var, Expr value, Expr index);

  static const IrNodeTy _node_type_ = IrNodeTy::Store;
};

/**
 * Allocate a buffer with the given type and size. The buffer lives for at most the duration of the body statement,
 * within which it is freed.
 */
struct Alloc : public ExprNode<Alloc> {
  Var buffer_var;
  //! Dimensions of this buffer (as a multi-dimensional array).
  std::vector<Expr> extents;
  Expr condition;
  Expr body;

  Alloc() : ExprNode(Type()) {}

  static Expr Make(Var buffer_var, Type type, const std::vector<Expr>& extents, Expr condition, Expr body);

  int32_t ConstantAllocationSize() const;
  static int32_t ConstantAllocationSize(const std::string& name, const std::vector<Expr>& extents);

  static const IrNodeTy _node_type_ = IrNodeTy::Alloc;
};

/**
 * Free the resources associated with the given buffer.
 */
struct Free : public ExprNode<Free> {
  Var var;

  Free() : ExprNode(Type()) {}

  static Expr Make(Var var);

  static const IrNodeTy _node_type_ = IrNodeTy::Free;
};

struct IfThenElse : public ExprNode<IfThenElse> {
  Expr condition;
  Expr true_case;
  Expr false_case;

  IfThenElse(Expr condition, Expr true_case, Expr false_case)
      : ExprNode(Type()), condition(condition), true_case(true_case), false_case(false_case) {
    CHECK(condition.defined());
    CHECK(true_case.defined());
  }

  static Expr Make(Expr condition, Expr true_case, Expr false_case);

  static const IrNodeTy _node_type_ = IrNodeTy::IfThenElse;
};

enum class ForType : int {
  //! Serial execution.
  Serial = 0,
  //! Parallel execution.
  Parallel = 1,
  //! Vector SIMD loop annotation.
  Vectorized = 2,
  //! Unroll annotation.
  Unrolled = 3,
};

struct For : public ExprNode<For> {
  //! The loop variable.
  Expr loop_var;
  //! The minimum value of the iteration.
  Expr min;
  //! The extent of the iteration.
  Expr extent;
  //! The type of the for loop.
  ForType for_type;

  Expr body;

  DeviceAPI device_api;

  For(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Expr body);

  static Expr Make(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Expr body);

  static const IrNodeTy _node_type_ = IrNodeTy::For;
};

//! Polyhedral forloop, which condition is more complex than the normal `For`.
struct PolyFor : public ExprNode<PolyFor> {
  //! The iterator variable.
  Var iterator;
  // Initial value of the iterator.
  Expr init;
  //! The condition to continue the loop.
  Expr condition;
  //! Increase the iterator.
  Expr inc;
  //! The forloop body.
  Expr body;

  ForType for_type;
  DeviceAPI device_api;

  PolyFor() : ExprNode(Type()) {}

  static Expr Make(
      Var iterator, Expr init_val, Expr condition, Expr inc, ForType for_type, DeviceAPI device_api, Expr body);

  static const IrNodeTy _node_type_ = IrNodeTy::PolyFor;
};

struct Module : public ExprNode<Module> {
  Module(Type t) : ExprNode<Module>(t) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Module;
};

struct Block : public ExprNode<Block> {
  std::vector<Expr> stmts;

  Block() : ExprNode(Type()) {}

  static Expr Make(const std::vector<Expr>& stmts);

  static const IrNodeTy _node_type_ = IrNodeTy::Block;
};

class _Range_;
class Range : public IrNodeRef {
 public:
  Range() = default;
  Range(IrNodeRef n) : IrNodeRef(n) {}
  Range(_Range_* n);
  _Range_* operator->() const { return get()->As<_Range_>(); }
};

class _Range_ : public IrNode {
 public:
  //! Begin of the range.
  Expr min;
  //! Extent of the range.
  Expr extent;

  _Range_() = default;
  _Range_(Expr min, Expr extent) : min(min), extent(extent) {}
  IrNodeTy node_type() const override { return _node_type_; }
  void Accept(IRVisitor* v) const override;

  static Range Make(Expr min, Expr extent) {
    auto node    = common::make_shared<_Range_>();
    node->min    = min;
    node->extent = extent;
    return Range(node);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::_Range_;
};

enum class IterVarType : int {
  /**
   * \brief Data parallel iteration.
   * This normally corresponds to axis of Tensor.
   * Allow all IterVar manipulations.
   *
   * \note This does't mean the loop have to be executed in parallel fashion.
   */
  kDataPar = 0,
  /**
   * \brief The IterVar itself is a thread-index of a fixed thread launching group.
   * \note This is already assumed to be parallized.
   *
   * Disallow: split/fuse/vectorize/parallel
   */
  kThreadIndex = 1,
  /**
   * \brief Communicative reduction.
   * \note Cannot be directly parallelized.
   *
   * Disallow: parallel/vectorize
   */
  kCommReduce = 2,
  /**
   * \brief Serial loops with loop carry dependency, the iteration must execute in order. Cannot be re-ordered.
   *
   * Disallow: reorder/parallel/vectorize.
   */
  kOrdered = 3,
  /**
   * \brief The loop is unrolled.
   */
  kUnrolled = 5,
  /**
   * \brief The loop is vectorized.
   */
  kVectorized = 6,
  /**
   * \brief The loop is parallelized.
   */
  kParallelized = 7,
};

class _IterVar_;
class IterVar : public IrNodeRef {
 public:
  IterVar() = default;
  IterVar(IrNodeRef n) : n_(n) {}
  _IterVar_* operator->() { return n_.As<_IterVar_>(); }
  const _IterVar_* operator->() const { return n_.As<_IterVar_>(); }

 private:
  IrNodeRef n_;
};

/**
 * An iteration variable representing an iteration over a one-dimensional interval.
 */
class _IterVar_ : public IrNode {
 public:
  //! The domain of the iteration.
  Range dom;
  //! The looping variable.
  Var var;
  //! The type of the IterVar.
  IterVarType iter_type;
  //! Additional tag on the iteration variable.
  std::string thread_tag;

  //! Create a new instance of IterVar.
  static IterVar Make(Range dom, Var var, IterVarType iter_type, const std::string& thread_tag = "");

  void Accept(IRVisitor* v) const override;
  IrNodeTy node_type() const override { return _node_type_; }

  static const IrNodeTy _node_type_ = IrNodeTy::_IterVar_;
};

static IterVar thread_axis(Range dom, const std::string& tag) {
  return _IterVar_::Make(dom, Var(tag), IterVarType::kThreadIndex, tag);
}
static IterVar reduce_axis(Range dom, const std::string& name) {
  return _IterVar_::Make(dom, Var(name), IterVarType::kCommReduce);
}

/**
 * A builder to construct any IR node.
 */
struct Builder {
  template <typename IRType, typename... Args>
  Expr MakeExpr(Args... args) {
    return IRType::Make(args...);
  }
};

}  // namespace ir

// Expose the following to cinn namespace for easier usage.
// @{
using ir::Expr;
using ir::Var;
// @}

}  // namespace cinn

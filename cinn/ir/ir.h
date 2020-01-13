#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cinn/ir/node.h"
#include "cinn/ir/type.h"

namespace cinn {
namespace ir {

/**
 * Cast a node to another type, can't change the width.
 */
struct Cast : public UnaryOpNode<Cast> {
  Cast(Type t, Expr v) : UnaryOpNode<Cast>(t, v) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Cast;
};

/**
 * The sum of two expressions.
 */
struct Add : public BinaryOpNode<Add> {
  Add(Expr a, Expr b) : BinaryOpNode<Add>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Add;
};

/**
 * The difference of two expressions.
 */
struct Sub : public BinaryOpNode<Sub> {
  Sub(Expr a, Expr b) : BinaryOpNode<Sub>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Sub;
};

/**
 * The product of two expressions.
 */
struct Mul : public BinaryOpNode<Mul> {
  Mul(Expr a, Expr b) : BinaryOpNode<Mul>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Mul;
};

/**
 * The ratio of two expressions.
 */
struct Div : public BinaryOpNode<Div> {
  Div(Expr a, Expr b) : BinaryOpNode<Div>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Div;
};

/**
 * The mod of two expressions.
 */
struct Mod : public BinaryOpNode<Mod> {
  Mod(Expr a, Expr b) : BinaryOpNode<Mod>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Mod;
};

/**
 * The lesser of two expressions.
 */
struct Min : public UnaryOpNode<Min> {
  Min(Expr v) : UnaryOpNode<Min>(v.type(), v) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Min;
};

/**
 * The larger of two expressions.
 */
struct Max : public UnaryOpNode<Max> {
  Max(Expr v) : UnaryOpNode<Max>(v.type(), v) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Max;
};

/**
 * Tell whether the first expression equals to the second expression.
 */
struct EQ : public BinaryOpNode<EQ> {
  EQ(Expr a, Expr b) : BinaryOpNode<EQ>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::EQ;
};

/**
 * Tell whether the first expression not equals to the second expression.
 */
struct NE : public BinaryOpNode<NE> {
  NE(Expr a, Expr b) : BinaryOpNode<NE>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::NE;
};

/**
 * Tell whether the first expression is lower than the second expression.
 */
struct LT : public BinaryOpNode<LT> {
  LT(Expr a, Expr b) : BinaryOpNode<LT>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::LT;
};

/**
 * Tell whether the first expression is no larger than the second expression.
 */
struct LE : public BinaryOpNode<LE> {
  LE(Expr a, Expr b) : BinaryOpNode<LE>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::LE;
};

/**
 * Tell whether the first expression is larger than the second expression.
 */
struct GT : public BinaryOpNode<GT> {
  GT(Expr a, Expr b) : BinaryOpNode<GT>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::GT;
};

/**
 * Tell whether the first expression is not less than the second expression.
 */
struct GE : public BinaryOpNode<GE> {
  GE(Expr a, Expr b) : BinaryOpNode<GE>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::GE;
};

/**
 * Logical and.
 */
struct And : public BinaryOpNode<And> {
  And(Expr a, Expr b) : BinaryOpNode<And>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::And;
};

/**
 * Logical or.
 */
struct Or : public BinaryOpNode<Or> {
  Or(Expr a, Expr b) : BinaryOpNode<Or>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Or;
};

/**
 * Logical not.
 */
struct Not : public BinaryOpNode<Not> {
  Not(Expr a, Expr b) : BinaryOpNode<Not>(a.type(), a, b) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Not;
};

struct Call : public ExprNode<Call> {
  Call(Type t) : ExprNode<Call>(t) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Call;
};

struct Variable : public ExprNode<Variable> {
  std::string name;

  Variable(const std::string& name, Type type) : ExprNode<Variable>(type), name(name) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Variable;
};

struct IfThenElse : public StmtNode<IfThenElse> {
  Expr condition;
  Expr true_case;
  Expr false_case;

  IfThenElse(Expr condition, Expr true_case, Expr false_case)
      : condition(condition), true_case(true_case), false_case(false_case) {
    CHECK(condition.defined());
    CHECK(true_case.defined());
  }

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

struct For : public StmtNode<For> {
  //! The loop variable.
  Expr loop_var;
  //! The minimum value of the iteration.
  Expr min;
  //! The extent of the iteration.
  Expr extent;
  //! The type of the for loop.
  ForType for_type;

  Stmt body;

  DeviceAPI device_api;

  For(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Stmt body) {
    this->min = std::move(min);
    this->extent = std::move(extent);
    this->for_type = std::move(for_type);
    this->device_api = device_api;
    this->body = std::move(body);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::For;
};

struct Module : public ExprNode<Module> {
  Module(Type t) : ExprNode<Module>(t) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Module;
};

struct Block : public StmtNode<Block> {
  std::vector<Stmt> stmts;

  Block() = default;

  static Stmt make(const std::vector<Stmt>& stmts) {
    auto node = std::make_shared<Block>();
    node->stmts = stmts;
    return Stmt(node);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::Block;
};

/**
 * A builder to construct any IR node.
 */
struct Builder {
  template <typename IRType, typename... Args>
  Expr make(Args... args) {
    return Expr(std::make_shared<IRType>(args...));
  }
};

}  // namespace ir
}  // namespace cinn

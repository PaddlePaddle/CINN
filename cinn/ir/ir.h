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

  static Expr Make(Type t, Expr v);

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

  static const IrNodeTy _node_type_ = IrNodeTy::Call;
};

struct Variable : public ExprNode<Variable> {
  std::string name;

  Variable(const std::string& name, Type type) : ExprNode<Variable>(type), name(name) {}

  static Expr Make(const std::string& name, Type type);

  static const IrNodeTy _node_type_ = IrNodeTy::Variable;
};

struct IfThenElse : public StmtNode<IfThenElse> {
  Expr condition;
  Stmt true_case;
  Stmt false_case;

  IfThenElse(Expr condition, Stmt true_case, Stmt false_case)
      : condition(condition), true_case(true_case), false_case(false_case) {
    CHECK(condition.defined());
    CHECK(true_case.defined());
  }

  static Stmt Make(Expr condition, Stmt true_case, Stmt false_case);

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

  For(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Stmt body);

  static Stmt Make(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Stmt body);

  static const IrNodeTy _node_type_ = IrNodeTy::For;
};

struct Module : public ExprNode<Module> {
  Module(Type t) : ExprNode<Module>(t) {}

  static const IrNodeTy _node_type_ = IrNodeTy::Module;
};

struct Block : public StmtNode<Block> {
  std::vector<Stmt> stmts;

  Block() = default;

  static Stmt Make(const std::vector<Stmt>& stmts);

  static const IrNodeTy _node_type_ = IrNodeTy::Block;
};

/**
 * A builder to construct any IR node.
 */
struct Builder {
  template <typename IRType, typename... Args>
  Expr MakeExpr(Args... args) {
    return IRType::Make(args...);
  }

  template <typename IRType, typename... Args>
  Stmt MakeStmt(Args... args) {
    return IRType::Make(args...);
  }
};

}  // namespace ir
}  // namespace cinn

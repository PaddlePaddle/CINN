#pragma once

#include <cinn/common/object.h>
#include <memory>
#include <string>

#include "cinn/common/shared.h"
#include "cinn/common/type.h"

namespace cinn {
namespace ir {
using common::Float;
using common::Int;
using common::Type;
using common::type_of;

class IrVisitor;

// clang-format off
#define NODETY_PRIMITIVE_TYPE_FOR_EACH(macro__) \
  macro__(IntImm)                               \
  macro__(UIntImm)                              \
  macro__(FloatImm)                             \

#define NODETY_OP_FOR_EACH(macro__) \
  macro__(Add)                      \
  macro__(Sub)                      \
  macro__(Mul)                      \
  macro__(Div)                      \
  macro__(Mod)                      \
  macro__(EQ)                       \
  macro__(NE)                       \
  macro__(LT)                       \
  macro__(LE)                       \
  macro__(GT)                       \
  macro__(GE)                       \
  macro__(And)                      \
  macro__(Or)                       \
  macro__(Not)                      \
  macro__(Min)                      \
  macro__(Max)                      \

#define NODETY_CONTROL_OP_FOR_EACH(macro__) \
  macro__(For)                              \
  macro__(Select)                           \
  macro__(IfThenElse)                       \
  macro__(Block)                            \
  macro__(Call)                             \
  macro__(Cast)                             \
  macro__(Module)                           \
  macro__(Variable)                         \
  macro__(Load)                             \
  macro__(Store)                            \
  macro__(Alloc)                            \
  macro__(Free)                             \
  macro__(_Range_)                            \
  macro__(_IterVar_)                            \
  macro__(_Buffer_)                            \
  macro__(_Tensor_)                            \

#define NODETY_FORALL(macro__)          \
  NODETY_PRIMITIVE_TYPE_FOR_EACH(__m)   \
  NODETY_OP_FOR_EACH(__m)               \
  NODETY_CONTROL_OP_FOR_EACH(__m)
// clang-format on

//! Define IrNodeTy
// @{
#define __m(x__) x__,
enum class IrNodeTy { kUnk = -1, NODETY_FORALL(__m) };
#undef __m
// @}

std::ostream& operator<<(std::ostream& os, IrNodeTy type);

/**
 * The base of all the nodes in the IR.
 */
class IrNode : public common::Object {
 public:
  IrNode() = default;
  IrNode(Type t) : type_(t) {}
  virtual ~IrNode() = default;

  virtual void Accept(IrVisitor* v) const = 0;
  virtual IrNodeTy node_type() const { return IrNodeTy ::kUnk; }
  virtual const Type& type() const { return type_; }
  void set_type(Type type) { type_ = type; }

  const char* type_info() const override { return __type_info__; }

 protected:
  constexpr static const char* __type_info__ = "IRNode";
  Type type_;
};

/**
 * A handle to store any IRNode.
 */
class IrNodeRef : public common::Shared<IrNode> {
 public:
  IrNodeRef() = default;
  IrNodeRef(const IrNodeRef& other) : Shared(other.p_) {}
  explicit IrNodeRef(IrNode* x) : Shared(x) {}

  virtual IrNodeTy node_type() const { return get()->node_type(); }

  template <typename T>
  const T* As() const {
    if (node_type() == T::_node_type_) return static_cast<const T*>(get());
    return nullptr;
  }
  template <typename T>
  T* As() {
    if (node_type() == T::_node_type_) return static_cast<T*>(get());
    return nullptr;
  }

  void operator=(const IrNodeRef& other) {
    *static_cast<Shared<IrNode>*>(this) = *static_cast<const Shared<IrNode>*>(&other);
  }

  IrNode* ptr() { return get(); }
  IrNode* ptr() const { return get(); }

  void Accept(IrVisitor* v) const { get()->Accept(v); }
};

template <typename T>
struct StmtNode : public IrNode {
  StmtNode() = default;

  void Accept(IrVisitor* v) const override;

  T* self() { return static_cast<T*>(this); }
  const T* const_self() const { return dynamic_cast<const T*>(this); }

  IrNodeTy node_type() const override { return T::_node_type_; }
};

template <typename T>
struct ExprNode : public IrNode {
  explicit ExprNode(Type t) : IrNode(t) {}

  void Accept(IrVisitor* v) const override;

  T* self() { return static_cast<T*>(this); }
  const T* const_self() const { return dynamic_cast<const T*>(this); }

  IrNodeTy node_type() const override { return T::_node_type_; }
};

struct IntImm : public ExprNode<IntImm> {
  int64_t value;

  IntImm(Type t, int64_t v) : ExprNode<IntImm>(t), value(v) {
    CHECK(t.is_int());
    CHECK(t.is_scalar());
    CHECK(t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::IntImm;
};

struct UIntImm : public ExprNode<UIntImm> {
  int64_t value;

  UIntImm(Type t, int64_t v) : ExprNode<UIntImm>(t), value(v) {
    CHECK(t.is_int());
    CHECK(t.is_scalar());
    CHECK(t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::UIntImm;
};

struct FloatImm : public ExprNode<FloatImm> {
  int value;

  FloatImm(Type t, float v) : ExprNode<FloatImm>(t), value(v) {
    CHECK(t.is_float());
    CHECK(t.is_scalar());
  }

  static const IrNodeTy _node_type_ = IrNodeTy::FloatImm;
};

class Var;
/**
 * An expression that represents some value or the result of some operations.
 */
struct Expr : public IrNodeRef {
 public:
  Expr() = default;
  Expr(const Expr& other) : IrNodeRef(other.ptr()) {}
  Expr(IrNode* p) : IrNodeRef(p) {}
  explicit Expr(const Var& var);

  //! Helper function to construct numeric constants of various types.
  // @{
  explicit Expr(int32_t x) : IrNodeRef(new IntImm(Int(32), x)) {}
  explicit Expr(int64_t x) : IrNodeRef(new IntImm(Int(64), x)) {}
  explicit Expr(float x) : IrNodeRef(new FloatImm(Float(32), x)) {}
  explicit Expr(double x) : IrNodeRef(new FloatImm(Float(64), x)) {}
  // @}

  Expr& operator=(const Expr& other) {
    *static_cast<IrNodeRef*>(this) = *static_cast<const IrNodeRef*>(&other);
    return *this;
  }

  const Type& type() const { return p_->type(); }
};

/**
 * An statement that doesn't have return value.
 */
struct Stmt : public IrNodeRef {
 public:
  Stmt() = default;

  Stmt(const Stmt& other) : IrNodeRef(other.ptr()) {}
  Stmt(IrNode* p) : IrNodeRef(p) {}
};

template <typename T>
struct UnaryOpNode : public ExprNode<T> {
  UnaryOpNode(Type type, Expr v) : ExprNode<T>(type), v(v) { CHECK(v.defined()); }

  // The single argument.
  Expr v;
};

template <typename T>
struct BinaryOpNode : public ExprNode<T> {
  BinaryOpNode(Type type, Expr a, Expr b) : ExprNode<T>(type), a(a), b(b) {
    CHECK(type.valid());
    CHECK(a.defined());
    CHECK(b.defined());
    CHECK_EQ(a.type(), b.type()) << "the two arguments' type not match";
  }

  //! The two arguments.
  Expr a, b;
};

enum class DeviceAPI {
  UNK,
  Host,
};

const DeviceAPI all_device_apis[] = {
    DeviceAPI::UNK,   //
    DeviceAPI::Host,  //
};

/**
 * An enum describing different address spaces to be used with Func::store_in.
 */
enum class MemoryType {
  Auto,

};

}  // namespace ir
}  // namespace cinn

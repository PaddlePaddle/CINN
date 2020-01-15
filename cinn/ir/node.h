#pragma once

#include <memory>
#include <string>

#include "cinn/ir/type.h"

namespace cinn {
namespace ir {

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
  macro__(Free)                            \

#define NODETY_FORALL(macro__)          \
  NODETY_PRIMITIVE_TYPE_FOR_EACH(__m)   \
  NODETY_OP_FOR_EACH(__m)               \
  NODETY_CONTROL_OP_FOR_EACH(__m)
// clang-format on

//! Define IrNodeTy
// @{
#define __m(x__) x__,
enum class IrNodeTy { NODETY_FORALL(__m) };
#undef __m
// @}

std::ostream& operator<<(std::ostream& os, IrNodeTy type);

class IRNode : public std::enable_shared_from_this<IRNode> {
 public:
  IRNode() = default;
  IRNode(Type t) : type_(t) {}
  virtual ~IRNode() = default;

  virtual void Accept(IrVisitor* v) const = 0;
  virtual IrNodeTy node_type() const = 0;
  virtual const Type& type() const { return type_; }

  std::shared_ptr<const IRNode> getptr() const { return shared_from_this(); }

 protected:
  Type type_;
};

/**
 * A handle to store any IRNode.
 */
class IRHandle : public std::enable_shared_from_this<IRHandle> {
 public:
  IRHandle() = default;
  IRHandle(IRHandle& other) : ptr_(other.ptr_) {}
  explicit IRHandle(IRNode* x) { ptr_.reset(x); }
  explicit IRHandle(const std::shared_ptr<IRNode>& x) { ptr_ = x; }

  IrNodeTy node_type() const { return ptr_->node_type(); }

  template <typename T>
  const T* As() const {
    if (node_type() == T::_node_type_) return static_cast<const T*>(ptr_.get());
    return nullptr;
  }
  template <typename T>
  T* As() {
    if (node_type() == T::_node_type_) return static_cast<T*>(ptr_.get());
    return nullptr;
  }

  bool defined() const { return ptr_.get(); }

  const std::shared_ptr<IRNode>& ptr() const { return ptr_; }
  void set_ptr(const std::shared_ptr<IRNode>& x) { ptr_ = x; }

  void Accept(IrVisitor* v) const { ptr_->Accept(v); }

 protected:
  std::shared_ptr<IRNode> ptr_{};
};

template <typename T>
struct StmtNode : public IRNode {
  StmtNode() = default;

  void Accept(IrVisitor* v) const override;

  T* self() { return static_cast<T*>(this); }
  const T* const_self() const { return static_cast<const T*>(this); }

  IrNodeTy node_type() const { return T::_node_type_; }
};

template <typename T>
struct ExprNode : public IRNode {
  explicit ExprNode(Type t) : IRNode(t) {}

  void Accept(IrVisitor* v) const override;

  T* self() { return static_cast<T*>(this); }
  const T* const_self() const { return static_cast<const T*>(this); }

  IrNodeTy node_type() const { return T::_node_type_; }
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

/**
 * An expression that represents some value or the result of some operations.
 */
struct Expr : public IRHandle {
 public:
  Expr() = default;
  Expr(const Expr& other) : IRHandle(other.ptr()) {}
  Expr(const std::shared_ptr<IRNode>& p) : IRHandle(p) {}
  void operator=(const std::shared_ptr<IRNode>& p) { ptr_ = p; }

  //! Helper function to construct numeric constants of various types.
  // @{
  explicit Expr(int32_t x) : IRHandle(std::make_shared<IntImm>(Int(32), x)) {}
  explicit Expr(int64_t x) : IRHandle(std::make_shared<IntImm>(Int(64), x)) {}
  explicit Expr(float x) : IRHandle(std::make_shared<IntImm>(Float(32), x)) {}
  explicit Expr(double x) : IRHandle(std::make_shared<IntImm>(Float(64), x)) {}
  // @}

  const Type& type() { return ptr_->type(); }
};

/**
 * An statement that doesn't have return value.
 */
struct Stmt : public IRHandle {
 public:
  Stmt() = default;

  Stmt(const Stmt& other) : IRHandle(other.ptr()) {}
  Stmt(const std::shared_ptr<IRNode>& p) : IRHandle(p) {}
  void operator=(const std::shared_ptr<IRNode>& p) { ptr_ = p; }
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
    CHECK(a.type() == b.type()) << "the two arguments' type not match";
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

#pragma once

#include <glog/logging.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/common/object.h"
#include "cinn/common/shared.h"
#include "cinn/common/type.h"

namespace cinn {
namespace ir {
using common::Float;
using common::Int;
using common::Type;
using common::type_of;

class IRVisitor;

// clang-format off
#define NODETY_PRIMITIVE_TYPE_FOR_EACH(macro__) \
  macro__(IntImm)                               \
  macro__(UIntImm)                              \
  macro__(FloatImm)                             \

#define NODETY_BINARY_OP_FOR_EACH(macro__) \
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
  macro__(Min)                      \
  macro__(Max)                      \

#define NODETY_UNARY_OP_FOR_EACH(macro__) \
  macro__(Minus)                          \
  macro__(Not)                            \

#define NODETY_OP_FOR_EACH(macro__) NODETY_BINARY_OP_FOR_EACH(macro__) NODETY_UNARY_OP_FOR_EACH(macro__)

#define NODETY_CONTROL_OP_FOR_EACH(macro__) \
  macro__(Cast)                             \
  macro__(For)                              \
  macro__(PolyFor)                          \
  macro__(Select)                           \
  macro__(IfThenElse)                       \
  macro__(Block)                            \
  macro__(Call)                             \
  macro__(Module)                           \
  macro__(_Var_)                            \
  macro__(Load)                             \
  macro__(Store)                            \
  macro__(Alloc)                            \
  macro__(Free)                             \
  macro__(_Range_)                          \
  macro__(_IterVar_)                        \
  macro__(_Buffer_)                         \
  macro__(_Tensor_)                         \
  macro__(_LoweredFunc_)                    \
  macro__(Let)                              \
  macro__(Reduce)                           \
  macro__(Ramp)                             \
  macro__(Broadcast)                        \

#define NODETY_FORALL(__m)              \
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

//! String representations for IrNodeTy.
// @{
#define __m(x__) #x__,
const std::vector<std::string> kIrNodeTyReprs({NODETY_FORALL(__m) "None"});
#undef __m
// @}

std::ostream& operator<<(std::ostream& os, IrNodeTy type);

/**
 * The base of all the nodes in the IR.
 */
class IrNode : public common::Object {
 public:
  IrNode() = default;
  explicit IrNode(Type t) : type_(t) {}
  virtual ~IrNode() = default;

  virtual void Accept(IRVisitor* v) const = 0;
  virtual IrNodeTy node_type() const { return IrNodeTy::kUnk; }
  virtual Type type() const { return type_; }
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

  virtual IrNodeTy node_type() const { return operator->()->node_type(); }

  template <typename T>
  const T* As() const {
    static_assert(std::is_base_of<IrNode, T>());
    CHECK(get()) << "IrNodeRef holds null";
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

  void Accept(IRVisitor* v) const { get()->Accept(v); }
};

struct Expr;

template <typename T>
struct ExprNode : public IrNode {
  //! The operands of this operator.
  std::vector<Expr> operands;

  ExprNode() : IrNode(Type()) {}
  explicit ExprNode(Type t) : IrNode(t) { set_type(t); }
  explicit ExprNode(int num_operands) { operands.resize(num_operands); }

  void Accept(IRVisitor* v) const override;

  T* self() { return static_cast<T*>(this); }
  const T* const_self() const { return dynamic_cast<const T*>(this); }

  //! Tell if this is an operator that takes one Expr as argument.
  bool is_unary_op() const { return operands.size() == 1; }
  //! Tell if this is an operator that takes two Exprs as arguments.
  bool is_binary_op() const { return operands.size() == 2; }

  //! Gather all the expression fields in this node for easier visit and mutate.
  virtual std::vector<Expr*> expr_fields() { return {}; }
  virtual std::vector<const Expr*> expr_fields() const { return {}; }

  virtual Expr Copy() const;

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
  double value;

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
  explicit Expr(IrNode* p) : IrNodeRef(p) {}
  explicit Expr(const Var& var);

  //! Helper function to construct numeric constants of various types.
  // @{
  explicit Expr(int32_t x) : IrNodeRef(new IntImm(Int(32), x)) {}
  explicit Expr(uint32_t x) : IrNodeRef(new IntImm(UInt(32), x)) {}
  explicit Expr(int64_t x) : IrNodeRef(new IntImm(Int(64), x)) {}
  explicit Expr(uint64_t x) : IrNodeRef(new IntImm(UInt(64), x)) {}
  explicit Expr(float x) : IrNodeRef(new FloatImm(Float(32), x)) {}
  explicit Expr(double x) : IrNodeRef(new FloatImm(Float(64), x)) {}
  // @}

  Expr& operator=(const Expr& other);

  // primitive types
  // @{
  int32_t as_int32() const;
  int64_t as_int64() const;
  float as_float() const;
  double as_double() const;
  // @}

  operator Var();

  Type type() const { return p_->type(); }
};

template <typename T>
struct UnaryOpNode : public ExprNode<T> {
  UnaryOpNode() { operands.resize(1); }
  UnaryOpNode(Type type, Expr v) : ExprNode<T>(type) {
    CHECK(v.defined());
    ExprNode<T>::operands.resize(1);
    this->v() = v;
  }

  Type type() const override {
    CHECK(v().defined());
    return v().type();
  }

  Expr& v() { return operands.front(); }
  const Expr& v() const { return operands.front(); }

  std::vector<Expr*> expr_fields() override { return {&v()}; }
  std::vector<const Expr*> expr_fields() const override { return {&v()}; }

  using ExprNode<T>::operands;
};

template <typename T>
struct BinaryOpNode : public ExprNode<T> {
  BinaryOpNode() { operands.resize(2); }
  BinaryOpNode(Type type, Expr a, Expr b) : ExprNode<T>(type) {
    CHECK(type.valid());
    CHECK(a.defined());
    CHECK(b.defined());
    operands.resize(2);
    this->a() = a;
    this->b() = b;
    CHECK_EQ(a.type(), b.type()) << "the two arguments' type not match";
  }

  Expr& a() { return operands[0]; }
  Expr& b() { return operands[1]; }
  const Expr& a() const { return operands[0]; }
  const Expr& b() const { return operands[1]; }

  Type type() const override {
    CHECK_EQ(a().type(), b().type());
    return a().type();
  }

  std::vector<Expr*> expr_fields() override { return {&a(), &b()}; }
  std::vector<const Expr*> expr_fields() const override { return {&a(), &b()}; }

  using ExprNode<T>::operands;
};

//! Zero in CINN type system.
Expr Zero(const Type& type);

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

template <typename T>
Expr ExprNode<T>::Copy() const {
  LOG(FATAL) << "Not Implemented";
  return Expr();
}

}  // namespace ir
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::ir::Expr> {
  size_t operator()(const cinn::ir::Expr& x) { return reinterpret_cast<size_t>(x.get()); }
};

}  // namespace std

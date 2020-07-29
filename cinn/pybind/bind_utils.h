#pragma once

#include <pybind11/pybind11.h>

#include <string>
#include <string_view>
#include <variant>

#include "cinn/common/cinn_value.h"
#include "cinn/common/shared.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/node.h"
#include "cinn/lang/tensor.h"

namespace py = pybind11;

namespace cinn::pybind {
using common::CINNValue;
using common::Shared;
using common::Type;
using ir::Expr;
using ir::ExprNode;

template <class... Ts>
struct Visitor : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
Visitor(Ts...)->Visitor<Ts...>;

using ExprOp   = std::variant<ir::IntImm,
                            ir::UIntImm,
                            ir::FloatImm,
                            ir::StringImm,
                            ir::Cast,
                            ir::Let,
                            ir::Reduce,
                            ir::Call,
                            ir::_Var_,
                            ir::Select,
                            ir::Load,
                            ir::Store,
                            ir::Alloc,
                            ir::Free,
                            ir::IfThenElse,
                            ir::For,
                            ir::PolyFor,
                            ir::Ramp,
                            ir::Broadcast,
                            ir::Power,
                            ir::Product,
                            ir::Sum,
                            ir::Block,
                            ir::_Module_,
                            ir::_Range_>;
using BinaryOp = std::variant<>;
using UnaryOp  = std::variant<>;

// hold CINNValue
using ValueVar = std::variant<int32_t, int64_t, float, ir::Var, ir::Expr, std::nullptr_t>;

inline ValueVar ConvertToVar(const CINNValue &value) {
  auto type_code = value.type_code();
  ValueVar var;
  if (type_code == CINNValue::type_code<int32_t>()) {
    var = static_cast<int32_t>(value);
  } else if (type_code == CINNValue::type_code<int64_t>()) {
    var = static_cast<int64_t>(value);
  } else if (type_code == CINNValue::type_code<float>()) {
    var = static_cast<float>(value);
  } else if (type_code == CINNValue::TypeCode<ir::Var>()) {
    var = ir::Var(value);
  } else if (type_code == CINNValue::TypeCode<ir::Expr>()) {
    var = ir::Expr(value);
  } else {
    var = nullptr;
  }

  return var;
}

template <typename T>
void DefineShared(py::module *m, std::string_view obj_name) {
  std::string name = "Shared" + std::string(obj_name);
  py::class_<Shared<T>> shared(*m, name.c_str());

  shared.def(py::init<>()).def(py::init<T *>()).def(py::init<const Shared<T> &>());
  //.def(py::init<Shared<T> &&>())k
  // TODO: .def("reset", &Shared<T>::Reset, py::arg("x") = nullptr)
  // TODO: .def("get", &Shared<T>::get, py::return_value_policy::reference);
}

template <typename NodeType>
void DefineExprNode(py::module *m, std::string_view node_name) {
  using ExprNodeT = ExprNode<NodeType>;

  std::string prefix{"ExprNode"};
  std::string name = prefix + std::string(node_name);
  py::class_<ExprNodeT, ir::IrNode> expr_node(*m, name.c_str(), py::module_local());
  expr_node.def(py::init<>())
      .def(py::init<Type>())
      .def(py::init<int>())
      .def("accept", &ExprNodeT::Accept)
      .def("operands_mutable", py::overload_cast<>(&ExprNodeT::operands))
      .def("operands_const", py::overload_cast<>(&ExprNodeT::operands, py::const_))
      .def("operand_mutable", py::overload_cast<int>(&ExprNodeT::operand), py::return_value_policy::reference)
      .def("operand_const", py::overload_cast<int>(&ExprNodeT::operand, py::const_), py::return_value_policy::reference)
      .def("copy", &ExprNodeT::Copy)
      .def("node_type", &ExprNodeT::node_type);
}

template <typename T>
void DefineExprNode1(py::module *m, T *node, std::string_view node_name) {
  using ExprNodeT = ExprNode<typename std::decay_t<decltype(*node)>::type>;
  std::string prefix{"ExprNode"};
  std::string name = prefix + std::string(node_name);
  py::class_<ExprNodeT, ir::IrNode> expr_node(*m, name.c_str());

  expr_node.def(py::init<>())
      .def(py::init<Type>())
      .def(py::init<int>())
      .def("accept", &ExprNodeT::Accept)
      .def("operands_mutable", py::overload_cast<>(&ExprNodeT::operands))
      .def("operands_const", py::overload_cast<>(&ExprNodeT::operands, py::const_))
      .def("operand_mutable", py::overload_cast<int>(&ExprNodeT::operand), py::return_value_policy::reference)
      .def("operand_const", py::overload_cast<int>(&ExprNodeT::operand, py::const_), py::return_value_policy::reference)
      .def("copy", &ExprNodeT::Copy)
      .def("node_type", &ExprNodeT::node_type);
}

template <typename NodeType>
void DefineBinaryOpNode(py::module *m, std::string_view node_name) {
  DefineExprNode<NodeType>(m, node_name);
  std::string prefix{"BinaryOpNode"};
  std::string name    = prefix + std::string(node_name);
  using BinaryOpNodeT = ir::BinaryOpNode<NodeType>;
  py::class_<BinaryOpNodeT, ir::ExprNode<NodeType>> binary_op_node(*m, name.c_str());
  binary_op_node.def(py::init<>())
      .def(py::init<Type, Expr, Expr>())
      .def("a_mutable", py::overload_cast<>(&BinaryOpNodeT::a), py::return_value_policy::reference)
      .def("a_const", py::overload_cast<>(&BinaryOpNodeT::a, py::const_), py::return_value_policy::reference)
      .def("b_mutable", py::overload_cast<>(&BinaryOpNodeT::b), py::return_value_policy::reference)
      .def("b_const", py::overload_cast<>(&BinaryOpNodeT::b, py::const_), py::return_value_policy::reference)
      .def("type", &BinaryOpNodeT::type)
      .def("expr_fields_mutable", py::overload_cast<>(&BinaryOpNodeT::expr_fields))
      .def("expr_fields_const", py::overload_cast<>(&BinaryOpNodeT::expr_fields, py::const_));
}

template <typename T>
void DefineBinaryOpNode1(py::module *m, T *node, std::string_view node_name) {
  using NodeType      = typename std::decay_t<decltype(*node)>::type;
  using BinaryOpNodeT = ir::BinaryOpNode<NodeType>;

  // DefineExprNode<NodeType>(m, node_name);

  if constexpr (std::is_same_v<T, ir::FracOp>) {
    node->def("is_constant", &ir::FracOp::is_constant).def("get_constant", &ir::FracOp::get_constant);
  }

  std::string prefix{"BinaryOpNode"};
  std::string name = prefix + std::string(node_name);
  py::class_<BinaryOpNodeT, ir::ExprNode<NodeType>> binary_op_node(*m, name.c_str());
  binary_op_node.def(py::init<>())
      .def(py::init<Type, Expr, Expr>())
      .def("a_mutable", py::overload_cast<>(&BinaryOpNodeT::a), py::return_value_policy::reference)
      .def("a_const", py::overload_cast<>(&BinaryOpNodeT::a, py::const_), py::return_value_policy::reference)
      .def("b_mutable", py::overload_cast<>(&BinaryOpNodeT::b), py::return_value_policy::reference)
      .def("b_const", py::overload_cast<>(&BinaryOpNodeT::b, py::const_), py::return_value_policy::reference)
      .def("type", &BinaryOpNodeT::type)
      .def("expr_fields_mutable", py::overload_cast<>(&BinaryOpNodeT::expr_fields))
      .def("expr_fields_const", py::overload_cast<>(&BinaryOpNodeT::expr_fields, py::const_));
}

template <typename NodeType>
void DefineUnaryOpNode(py::module *m, std::string_view node_name) {
  using UnaryOpNodeT = ir::UnaryOpNode<NodeType>;
  DefineExprNode<NodeType>(m, node_name);

  std::string name = "UnaryOpNode" + std::string(node_name);
  py::class_<UnaryOpNodeT, ir::ExprNode<NodeType>> unary_op_node(*m, name.c_str());
  unary_op_node.def(py::init<>())
      .def(py::init<Type, Expr>())
      .def("type", &UnaryOpNodeT::type)
      .def("v_mutable", py::overload_cast<>(&UnaryOpNodeT::v), py::return_value_policy::reference)
      .def("v_const", py::overload_cast<>(&UnaryOpNodeT::v, py::const_), py::return_value_policy::reference)
      .def("expr_fields_mutable", py::overload_cast<>(&UnaryOpNodeT::expr_fields))
      .def("expr_fields_const", py::overload_cast<>(&UnaryOpNodeT::expr_fields, py::const_))
      .def("operands_mutable", py::overload_cast<>(&UnaryOpNodeT::operands), py::return_value_policy::reference)
      .def("operands_const",
           py::overload_cast<>(&UnaryOpNodeT::operands, py::const_),
           py::return_value_policy::reference);
}

class ObjectWrapper : public Object {
 public:
  using Object::Object;

  const char *type_info() const override { PYBIND11_OVERLOAD_PURE(const char *, Object, type_info); }
};

class IrNodeWrapper : ir::IrNode {
 public:
  using ir::IrNode::IrNode;
  void Accept(ir::IRVisitor *v) const override { PYBIND11_OVERLOAD_PURE(void, ir::IrNode, Accept); }
};

class _Operation_Wrapper : ir::_Operation_ {
 public:
  const char *func_type() const override { PYBIND11_OVERLOAD_PURE(const char *, ir::_Operation_, func_type); }
};
}  // namespace cinn::pybind

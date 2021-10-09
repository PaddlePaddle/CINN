#include "cinn/frontend/cinn_builder.h"

#include <glog/logging.h>

#include <string>
#include <utility>
#include <vector>

#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

#define UNARY_OP_DEF(name__) Variable CinnBuilder::name__(const Variable& operand)
UNARY_OP_DEF(Exp) { return UnaryOp("exp", operand); }

UNARY_OP_DEF(Erf) { return UnaryOp("erf", operand); }

UNARY_OP_DEF(Sqrt) { return UnaryOp("sqrt", operand); }

UNARY_OP_DEF(Rsqrt) { return UnaryOp("rsqrt", operand); }

UNARY_OP_DEF(Log) { return UnaryOp("log", operand); }

UNARY_OP_DEF(Log2) { return UnaryOp("log2", operand); }

UNARY_OP_DEF(Log10) { return UnaryOp("log10", operand); }

UNARY_OP_DEF(Floor) { return UnaryOp("floor", operand); }

UNARY_OP_DEF(Ceil) { return UnaryOp("ceil", operand); }

UNARY_OP_DEF(Round) { return UnaryOp("round", operand); }

UNARY_OP_DEF(Trunc) { return UnaryOp("trunc", operand); }

UNARY_OP_DEF(Sin) { return UnaryOp("sin", operand); }

UNARY_OP_DEF(Cos) { return UnaryOp("cos", operand); }

UNARY_OP_DEF(Tan) { return UnaryOp("tan", operand); }

UNARY_OP_DEF(Sinh) { return UnaryOp("sinh", operand); }

UNARY_OP_DEF(Cosh) { return UnaryOp("cosh", operand); }

UNARY_OP_DEF(Tanh) { return UnaryOp("tanh", operand); }

UNARY_OP_DEF(Asin) { return UnaryOp("asin", operand); }

UNARY_OP_DEF(Acos) { return UnaryOp("acos", operand); }

UNARY_OP_DEF(Atan) { return UnaryOp("atan", operand); }

UNARY_OP_DEF(Asinh) { return UnaryOp("asinh", operand); }

UNARY_OP_DEF(Acosh) { return UnaryOp("acosh", operand); }

UNARY_OP_DEF(Atanh) { return UnaryOp("atanh", operand); }

UNARY_OP_DEF(IsNan) { return UnaryOp("isnan", operand); }

UNARY_OP_DEF(IsFinite) { return UnaryOp("isfinite", operand); }

UNARY_OP_DEF(IsInf) { return UnaryOp("isinf", operand); }

UNARY_OP_DEF(LogicalNot) { return UnaryOp("logical_not", operand); }

UNARY_OP_DEF(BitwiseNot) { return UnaryOp("bitwise_not", operand); }

UNARY_OP_DEF(Negative) { return UnaryOp("negative", operand); }

UNARY_OP_DEF(Sign) { return UnaryOp("sign", operand); }

UNARY_OP_DEF(Abs) { return UnaryOp("abs", operand); }

UNARY_OP_DEF(Identity) { return UnaryOp("identity", operand); }
#undef UNARY_OP_DEF

#define BINARY_OP_DEF(name__) Variable CinnBuilder::name__(const Variable& lhs, const Variable& rhs)
BINARY_OP_DEF(Dot) { return BinaryOp("matmul", lhs, rhs); }

BINARY_OP_DEF(Add) { return BinaryOp("elementwise_add", lhs, rhs); }

BINARY_OP_DEF(Sub) { return BinaryOp("substract", lhs, rhs); }

BINARY_OP_DEF(Mul) { return BinaryOp("elementwise_mul", lhs, rhs); }

BINARY_OP_DEF(Div) { return BinaryOp("divide", lhs, rhs); }

BINARY_OP_DEF(FloorDiv) { return BinaryOp("floor_divide", lhs, rhs); }

BINARY_OP_DEF(Mod) { return BinaryOp("mod", lhs, rhs); }

BINARY_OP_DEF(FloorMod) { return BinaryOp("floor_mod", lhs, rhs); }

BINARY_OP_DEF(Max) { return BinaryOp("max", lhs, rhs); }

BINARY_OP_DEF(Min) { return BinaryOp("min", lhs, rhs); }

BINARY_OP_DEF(Power) { return BinaryOp("power", lhs, rhs); }

BINARY_OP_DEF(LogicalAnd) { return BinaryOp("logical_and", lhs, rhs); }

BINARY_OP_DEF(LogicalOr) { return BinaryOp("logical_or", lhs, rhs); }

BINARY_OP_DEF(LogicalXor) { return BinaryOp("logical_xor", lhs, rhs); }

BINARY_OP_DEF(BitwiseAnd) { return BinaryOp("bitwise_and", lhs, rhs); }

BINARY_OP_DEF(BitwiseOr) { return BinaryOp("bitwise_or", lhs, rhs); }

BINARY_OP_DEF(BitwiseXor) { return BinaryOp("bitwise_xor", lhs, rhs); }

BINARY_OP_DEF(LeftShift) { return BinaryOp("left_shift", lhs, rhs); }

BINARY_OP_DEF(RightShift) { return BinaryOp("right_shift", lhs, rhs); }
#undef BINARY_OP_DEF

Variable CinnBuilder::Concat(const Variable& lhs, const Variable& rhs, int axis) {
  Instruction instr("concat", {lhs, rhs});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::Conv(const Variable& lhs,
                           const Variable& rhs,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& dilations,
                           int groups,
                           const std::string& data_format,
                           const std::string& padding_algorithm) {
  Instruction instr("conv2d");
  instr.SetInputs({lhs, rhs});
  instr.SetAttr("strides", strides);
  instr.SetAttr("paddings", paddings);
  instr.SetAttr("dilations", dilations);
  instr.SetAttr("groups", groups);
  instr.SetAttr("data_format", data_format);
  instr.SetAttr("padding_algorithm", padding_algorithm);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::Compare(ComparisonKind kind, const Variable& lhs, const Variable& rhs) {
  switch (kind) {
    case ComparisonKind::kEq:
      return BinaryOp("equal", lhs, rhs);
    case ComparisonKind::kNe:
      return BinaryOp("not_equal", lhs, rhs);
    case ComparisonKind::kGe:
      return BinaryOp("greater_equal", lhs, rhs);
    case ComparisonKind::kGt:
      return BinaryOp("greater", lhs, rhs);
    case ComparisonKind::kLe:
      return BinaryOp("less_equal", lhs, rhs);
    case ComparisonKind::kLt:
      return BinaryOp("less", lhs, rhs);
    default:
      LOG(FATAL) << "unknown comparison kind";
  }
}

Variable CinnBuilder::Reduce(ReduceKind kind, const Variable& operand, const std::vector<int>& dim, bool keep_dim) {
  switch (kind) {
    case ReduceKind::kSum:
      return ReduceOp("reduce_sum", operand, dim, keep_dim);
    case ReduceKind::kProd:
      return ReduceOp("reduce_prod", operand, dim, keep_dim);
    case ReduceKind::kMax:
      return ReduceOp("reduce_max", operand, dim, keep_dim);
    case ReduceKind::kMin:
      return ReduceOp("reduce_min", operand, dim, keep_dim);
    default:
      LOG(FATAL) << "unknown reduction kind";
  }
}

Variable CinnBuilder::BroadcastTo(const Variable& operand,
                                  const std::vector<int>& out_shape,
                                  const std::vector<int>& broadcast_axes) {
  Instruction instr("broadcast_to");
  instr.SetInputs({operand});
  instr.SetAttr("out_shape", out_shape);
  instr.SetAttr("broadcast_axes", broadcast_axes);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::Reshape(const Variable& operand, const std::vector<int>& shape) {
  Instruction instr("reshape", {operand});
  instr.SetAttr("shape", shape);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::Slice(const Variable& operand,
                            const std::vector<int>& axes,
                            const std::vector<int>& starts,
                            const std::vector<int>& ends) {
  Instruction instr("slice", {operand});
  instr.SetAttr("axes", axes);
  instr.SetAttr("starts", starts);
  instr.SetAttr("ends", ends);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::Select(const Variable& condition, const Variable& true_value, const Variable& false_value) {
  Instruction instr("select", {condition, true_value, false_value});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::Reverse(const Variable& operand, const std::vector<int>& axis) {
  Instruction instr("reverse", {operand});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::UnaryOp(const std::string& op_type, const Variable& operand) {
  Instruction instr(op_type, {operand});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs) {
  Instruction instr(op_type, {lhs, rhs});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::ReduceOp(const std::string& op_type,
                               const Variable& operand,
                               const std::vector<int>& dim,
                               bool keep_dim) {
  Instruction instr(op_type, {operand});
  instr.SetAttr("dim", dim);
  instr.SetAttr("keep_dim", keep_dim);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

}  // namespace frontend
}  // namespace cinn

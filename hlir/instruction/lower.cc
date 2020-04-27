#include "hlir/instruction/lower.h"
#include <glog/logging.h>
#include <unordered_set>
#include "cinn/cinn.h"
#include "cinn/common/common.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"
#include "hlir/instruction/instructions.h"
#include "hlir/instruction/primitive/dot.h"
#include "hlir/instruction/shape.h"

namespace hlir {
namespace instruction {
using cinn::ir::_Tensor_;
using cinn::ir::Expr;
using cinn::ir::Tensor;
using cinn::ir::Var;
using cinn::lang::Compute;

std::unique_ptr<cinn::lang::Module> Lower(const Module& module) { return std::unique_ptr<cinn::lang::Module>(); }

Tensor BinaryImpl(
    const Tensor& a, const Tensor& b, const std::string& name, bool is_inline, std::function<Expr(Expr, Expr)> op) {
  CHECK(a.defined());
  CHECK(b.defined());

  int ndims = a->shape.size();
  auto axis = cinn::common::GenDefaultAxis(ndims);

  std::vector<Expr> shape;
  Tensor out_tensor;
  switch (ndims) {
    case 1:
      out_tensor = Compute(
          a->shape, [a, b, op](Var i) -> Expr { return op(a(i), b(i)); }, name);
      break;
    case 2:
      out_tensor = Compute(
          a->shape, [a, b, op](Var i, Var j) -> Expr { return op(a(i, j), b(i, j)); }, name);
      break;
    case 3:
      out_tensor = Compute(
          a->shape, [a, b, op](Var i, Var j, Var k) -> Expr { return op(a(i, j, k), b(i, j, k)); }, name);
      break;
    case 4:
      out_tensor = Compute(
          a->shape, [a, b, op](Var i, Var j, Var k, Var m) -> Expr { return op(a(i, j, k, m), b(i, j, k, m)); }, name);
      break;
    default:
      NOT_IMPLEMENTED
  }

  if (!is_inline) out_tensor->WithBuffer();

  return out_tensor;
}

/**
 * Implement the Call.
 * @param fn_name The call target.
 * @param args The readonly arguments.
 * @param shapes The shapes of the return tensors.
 * @param tensor_names The names of the return tensors.
 * @return The expression of the call.
 */
std::vector<Tensor> CallImpl(const std::string& fn_name,
                             const std::vector<Expr>& args,
                             const std::vector<Shape>& shapes,
                             const std::vector<std::string>& tensor_names,
                             const std::vector<cinn::common::Type>& types) {
  CHECK_EQ(shapes.size(), tensor_names.size());
  std::vector<cinn::lang::ReturnType> return_types(shapes.size());
  for (int i = 0; i < shapes.size(); i++) {
    return_types[i].name = tensor_names[i];
    return_types[i].dims = shapes[i].ToCinnShape();
    return_types[i].type = types[i];
  }

  return cinn::lang::Call(fn_name, args, return_types);
}

void ComputationLower::LowerBinary(const Instruction* instr) {
  auto* a_instr = instr->operand(0);
  auto* b_instr = instr->operand(1);
  CHECK_EQ(a_instr->shape(), b_instr->shape());

  auto a_expr = scope_.Lookup(a_instr);
  auto b_expr = scope_.Lookup(b_instr);
  CHECK(a_expr.defined());
  CHECK(b_expr.defined());

  auto* a_expr_tensor = a_expr.As<cinn::ir::_Tensor_>();
  auto* b_expr_tensor = b_expr.As<cinn::ir::_Tensor_>();

  CHECK(a_expr_tensor);
  CHECK(b_expr_tensor);

  Tensor C;
  switch (instr->instr_code()) {
    case InstrCode::Add:
      C = BinaryImpl(
          Tensor(a_expr_tensor), Tensor(b_expr_tensor), instr->programable_id(), instr->inlined(), [](Expr a, Expr b) {
            return a + b;
          });
      break;
    case InstrCode::Sub:
      C = BinaryImpl(
          Tensor(a_expr_tensor), Tensor(b_expr_tensor), instr->programable_id(), instr->inlined(), [](Expr a, Expr b) {
            return a - b;
          });
    case InstrCode::Mul:
      C = BinaryImpl(
          Tensor(a_expr_tensor), Tensor(b_expr_tensor), instr->programable_id(), instr->inlined(), [](Expr a, Expr b) {
            return a * b;
          });
    case InstrCode::Div:
      C = BinaryImpl(
          Tensor(a_expr_tensor), Tensor(b_expr_tensor), instr->programable_id(), instr->inlined(), [](Expr a, Expr b) {
            return a / b;
          });

    default:
      NOT_IMPLEMENTED
  }

  CHECK(C.defined());
  scope_.Insert(instr, C);
}

void ComputationLower::LowerParameter(const Instruction* instr) {
  CHECK(instr->type().valid());
  CHECK_EQ(instr->instr_code(), InstrCode::Parameter);
  Tensor placeholder = cinn::lang::CreatePlaceHolder(
      instr->shape().ToCinnShape(), instr->type(), instr->As<ParameterInstruction>()->name());
  scope_.Insert(instr, placeholder);
}

void ComputationLower::LowerInstruction(const Instruction* instr) {
  if (scope_.Lookup(instr).defined()) return;

  switch (instr->instr_code()) {
    case InstrCode::Add:
    case InstrCode::Sub:
    case InstrCode::Mul:
    case InstrCode::Div:
      LowerBinary(instr);
      break;
    case InstrCode::Dot:
      LowerDot(instr);
      break;
    case InstrCode::Parameter:
      LowerParameter(instr);
      break;
    case InstrCode::Call:
      LowerCall(instr);
      break;

    case InstrCode::Tuple:
      LowerTuple(instr);
      break;
    case InstrCode::TupleGet:
      LowerTupleGet(instr);
      break;

    default:
      NOT_IMPLEMENTED
  }
}

void ComputationLower::LowerDot(const Instruction* instr) {
  auto* ai = instr->operand(0);
  auto* bi = instr->operand(1);
  Expr av  = scope_.Lookup(ai);
  Expr bv  = scope_.Lookup(bi);
  CHECK(av.defined());
  CHECK(bv.defined());
  CHECK(av.as_tensor());
  CHECK(bv.as_tensor());

  primitive::DotImpl dot_impl(ctx_);
  auto out = dot_impl(av.as_tensor_ref(), bv.as_tensor_ref(), instr->programable_id());
  scope_.Insert(instr, out);
}

void ComputationLower::LowerTuple(const Instruction* instr) {}

void ComputationLower::LowerTupleGet(const Instruction* instr) {
  auto* tuple_get = instr->As<TupleGet>();
  if (tuple_get->tuple()->call()) {
    auto it = call_to_ret_vals_.find(tuple_get->tuple()->call());
    CHECK(it != call_to_ret_vals_.end());
    scope_.Insert(instr, it->second[tuple_get->offset()]);
  } else if (!tuple_get->tuple()->items().empty()) {
    auto* key = tuple_get->tuple()->items()[tuple_get->offset()];
    auto expr = scope_.Lookup(key);
    CHECK(expr.defined());
    scope_.Insert(instr, expr);
  } else {
    NOT_IMPLEMENTED
  }
}

Expr ComputationLower::operator()(const Computation* computation) {
  for (auto& instr : computation->instructions()) {
    LowerInstruction(instr.get());
  }

  // collect parameters
  std::vector<Var> vars = computation->GetVars();

  std::vector<Tensor> tensors;
  std::unordered_set<std::string> tensor_names;

  /*
   * Both the parameters and constants are tensor parameters of the lowered function. The constants will be extracted
   * into the global scope in process and shared across the whole module, and pass as normal readonly buffers in the
   * lowered functions.
   */

  auto tensor_add = [&](_Tensor_* tensor) {
    if (!tensor_names.count(tensor->name)) {
      tensors.push_back(Tensor(tensor));
      tensor_names.insert(tensor->name);
    }
  };

  for (auto& instr : computation->instructions()) {
    auto expr = scope_.Lookup(instr.get());
    if (!expr.defined()) continue;
    auto* expr_tensor = expr.As<_Tensor_>();

    if (expr_tensor && !expr_tensor->inlined()) {
      tensor_add(expr_tensor);
    }

    if (instr->As<CallInstruction>()) {
      auto it = call_to_ret_vals_.find(instr.get());
      if (it != call_to_ret_vals_.end()) {
        for (auto& tensor : it->second) {
          if (tensor.as_tensor()) {
            tensor_add(tensor.as_tensor());
          }
        }
      }
    }
  }

  auto fn = cinn::Lower(computation->name(), tensors, vars);
  return fn;
}

void ComputationLower::LowerCall(const Instruction* instr) {
  auto* call = instr->As<CallInstruction>();
  std::vector<Expr> args;
  for (int i = 0; i < instr->operand_count(); i++) {
    LowerInstruction(instr->operand(i));
    auto instr_expr = scope_.Lookup(instr->operand(i));
    CHECK(instr_expr.defined());
    args.push_back(instr_expr);
  }

  auto tensors =
      CallImpl(call->computation()->name(), args, call->ret_shapes(), call->ret_tensor_names(), call->ret_types());
  std::vector<Expr> arg_exprs;
  for (auto& tensor : tensors) arg_exprs.emplace_back(tensor);
  call_to_ret_vals_[call] = arg_exprs;
}

cinn::Module ModuleLower::operator()(const Module* module) {
  // TODO(Superjomn) Refine the target.
  cinn::Module cinn_module(module->name(), cinn::Target());
  for (auto& item : module->computations()) {
    Expr expr = LowerComputation(item.second.get());
    cinn_module.Append(cinn::ir::LoweredFunc(expr.As<cinn::ir::_LoweredFunc_>()));
  }
  return cinn_module;
}

cinn::Expr ModuleLower::LowerComputation(const Computation* computation) {
  Context context;
  ComputationLower lower(&scope_, &context);
  return lower(computation);
}

}  // namespace instruction
}  // namespace hlir

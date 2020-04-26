#include "hlir/instruction/lower.h"
#include <glog/logging.h>
#include "cinn/cinn.h"
#include "cinn/common/common.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"
#include "hlir/instruction/instructions.h"
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

Expr CallImpl(const std::string& fn_name, const std::vector<Expr>& args) {}

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
    case InstrCode::Call:
      break;

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
  switch (instr->instr_code()) {
    case InstrCode::Add:
    case InstrCode ::Sub:
    case InstrCode ::Mul:
    case InstrCode ::Div:
      LowerBinary(instr);
      break;
    case InstrCode ::Dot:
      LowerDot(instr);
      break;
    case InstrCode::Parameter:
      LowerParameter(instr);
      break;

    default:
      NOT_IMPLEMENTED
  }
}

void ComputationLower::LowerDot(const Instruction* instr) {}

Expr ComputationLower::operator()(const Computation* computation) {
  for (auto& instr : computation->instructions()) {
    LowerInstruction(instr.get());
  }

  // collect parameters
  std::vector<Var> vars = computation->GetVars();

  std::vector<Tensor> tensors;

  /*
   * Both the parameters and constants are tensor parameters of the lowered function. The constants will be extracted
   * into the global scope in process and shared across the whole module, and pass as normal readonly buffers in the
   * lowered functions.
   */

  for (auto& instr : computation->instructions()) {
    auto expr = scope_.Lookup(instr.get());
    CHECK(expr.defined());
    auto* expr_tensor = expr.As<_Tensor_>();

    if (expr_tensor && !expr_tensor->inlined()) {
      tensors.push_back(Tensor(expr_tensor));
    }
  }

  auto fn = cinn::Lower(computation->name(), tensors, vars);
  return fn;
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
  ComputationLower lower(&scope_);
  return lower(computation);
}

}  // namespace instruction
}  // namespace hlir

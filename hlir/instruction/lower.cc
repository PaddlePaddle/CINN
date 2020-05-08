#include "hlir/instruction/lower.h"

#include <glog/logging.h>
#include <glog/raw_logging.h>

#include <iostream>
#include <unordered_set>

#include "cinn/cinn.h"
#include "cinn/common/common.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"
#include "cinn/utils/string.h"
#include "hlir/instruction/instructions.h"
#include "hlir/instruction/primitive/binary.h"
#include "hlir/instruction/primitive/dot.h"
#include "hlir/instruction/shape.h"

namespace hlir {
namespace instruction {
using cinn::ir::_Tensor_;
using cinn::ir::Expr;
using cinn::ir::Tensor;
using cinn::ir::Var;
using cinn::lang::Compute;

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
  CHECK_GE(a_instr->shape().num_dims(), b_instr->shape().num_dims());

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
#define BINARY_OPR(key__, op__)                                                 \
  case InstrCode::key__:                                                        \
    C = primitive::BinaryImpl(                                                  \
        ctx_, [](Expr a, Expr b) { return a op__ b; }, instr->inlined())(       \
        Tensor(a_expr_tensor), Tensor(b_expr_tensor), instr->programable_id()); \
    break;

    BINARY_OPR(Add, +);
    BINARY_OPR(Sub, -);
    BINARY_OPR(Mul, *);
    BINARY_OPR(Div, /);

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

void ComputationLower::LowerTuple(const Instruction* instr) {
  // Tuple is just a placeholder for CINN, nothing need to do now.
}

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
  std::vector<Var> params = computation->CollecVars();

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

  auto fn = cinn::Lower(computation->name(), tensors, params);
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

cinn::Module ModuleLower::operator()(const Module* module, bool display_c_code) {
  std::cerr << "Lower get HLIR module:\n" << module->to_debug_string() << std::endl;

  // TODO(Superjomn) Refine the target.
  cinn::Module::Builder builder(module->name(), cinn::Target());

  // lower functions but main
  for (auto& item : module->computations()) {
    if (item.second.get() == module->entry_computation()) continue;
    Expr expr = LowerComputation(item.second.get());
    VLOG(2) << "HLIR lower get CINN function:\n" << expr;
    builder.AddFunction(cinn::ir::LoweredFunc(expr.As<cinn::ir::_LoweredFunc_>()));
  }
  // lower main function
  if (module->entry_computation()) {
    Expr expr = LowerComputation(module->entry_computation());
    builder.AddFunction(cinn::ir::LoweredFunc(expr.As<cinn::ir::_LoweredFunc_>()));
  }

  cinn::Module cinn_module = builder.Build();

  if (display_c_code) {
    cinn::backends::CodeGenC codegen_c(cinn::common::DefaultHostTarget());
    codegen_c.SetInlineBuiltinCodes(false);
    RAW_LOG(INFO, "C sample code:");
    std::cerr << codegen_c.Compile(cinn_module, cinn::backends::CodeGenC::OutputKind::CImpl) << std::endl;
  }

  return cinn_module;
}

cinn::Expr ModuleLower::LowerComputation(const Computation* computation) {
  Context context;
  ComputationLower lower(&scope_, &context);
  return lower(computation);
}

cinn::lang::Module Lower(const Module& module, bool display_c_code) { return ModuleLower()(&module, display_c_code); }

}  // namespace instruction
}  // namespace hlir

#include "module_lower.h"

#include <glog/logging.h>
#include <glog/raw_logging.h>

#include <iostream>
#include <unordered_set>

#include "cinn/cinn.h"
#include "cinn/common/common.h"
#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/primitive/binary.h"
#include "cinn/hlir/instruction/primitive/conv.h"
#include "cinn/hlir/instruction/primitive/dot.h"
#include "cinn/hlir/instruction/primitive/elementwise.h"
#include "cinn/hlir/instruction/primitive/use_primitives.h"
#include "cinn/hlir/instruction/shape.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace instruction {
using cinn::ir::_Tensor_;
using cinn::ir::Expr;
using cinn::ir::Tensor;
using cinn::ir::Var;
using cinn::lang::Compute;

void ComputationLower::LowerInstruction(const Instruction* instr) {
  // Avoid duplicate lower.
  if (scope_.Lookup(instr).defined()) return;

  auto lower = LowerImplRegistry::Global().Create(instr->instr_code(), instr->lower_kind());
  CHECK(lower) << "No lower impl found for Instruction [" << instr->instr_code() << "] in lower mode ["
               << instr->lower_kind() << "]";
  lower->Run(instr, ctx_, &scope_, this);
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

void ComputationLower::LowerConv(const Instruction* instr) {
  auto* op = instr->As<Conv>();

  auto I_expr = scope_.Lookup(instr->operand(0));
  auto W_expr = scope_.Lookup(instr->operand(1));
  CHECK(I_expr.defined());
  CHECK(W_expr.defined());

  auto tensor = primitive::Conv2dNCHW(
      I_expr.as_tensor_ref(), W_expr.as_tensor_ref(), op->pad_h(), op->pad_w(), op->stride_h(), op->stride_w());
  if (!instr->inlined()) {
    tensor->WithBuffer();
  }
  scope_.Insert(instr, tensor);
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
}  // namespace cinn

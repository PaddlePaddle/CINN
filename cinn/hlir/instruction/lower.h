#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/hlir/instruction/computation.h"
#include "cinn/hlir/instruction/instruction.h"
#include "cinn/hlir/instruction/module.h"
#include "cinn/hlir/instruction/scope.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/module.h"

namespace cinn {
namespace hlir {
namespace instruction {

/**
 * Lower the HLIR module to CINN module.
 */
cinn::lang::Module Lower(const Module& module, bool display_c_code = false);

/**
 * Lower an HLIR Module to CINN Module.
 */
class ModuleLower {
 public:
  ModuleLower() = default;

  //! Lower a module.
  cinn::Module operator()(const Module* module, bool display_C_code = false);

 private:
  //! Lower a computation.
  cinn::Expr LowerComputation(const Computation* computation);

 private:
  //! parent scope, the parent of all the computations' local scopes.
  Scope scope_;
};

/**
 * Lower an HLIR Computation to CINN expression.
 */
class ComputationLower {
 public:
  explicit ComputationLower(Scope* parent_scope, Context* ctx) : scope_(parent_scope), ctx_(ctx) {}

  /**
   * Lower a HLIR computation and get a CINN LoweredFunc expression.
   * @param computation The computation to lower.
   * @return The equivalent CINN LoweredFunc expression.
   */
  cinn::Expr operator()(const Computation* computation);

 private:
  void LowerInstruction(const Instruction* instr);

  void LowerDot(const Instruction* instr);

  void LowerCall(const Instruction* instr);

  void LowerCustomCall(const Instruction* instr);

  void LowerBinary(const Instruction* instr);

  void LowerParameter(const Instruction* instr);

  void LowerTuple(const Instruction* instr);

  void LowerTupleGet(const Instruction* instr);

  void LowerUnary(const Instruction* instr);

 private:
  std::unordered_map<const Instruction*, std::vector<cinn::Expr>> call_to_ret_vals_;
  Scope scope_;
  Context* ctx_{};
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

#pragma once

#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/lower_impl.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

class ParameterLowerImpl : public LowerImplBase {
 public:
  explicit ParameterLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(const Instruction* instr, Context* context, Scope* scope, ComputationLower* lower) override {
    CHECK(instr->type().valid());
    CHECK_EQ(instr->instr_code(), InstrCode::Parameter);
    ir::Tensor placeholder = cinn::lang::CreatePlaceHolder(
        instr->shape().ToCinnShape(), instr->type(), instr->As<ParameterInstruction>()->name());
    scope->Insert(instr, placeholder);
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

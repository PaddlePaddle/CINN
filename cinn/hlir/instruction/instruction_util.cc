#include "cinn/hlir/instruction/instruction_util.h"

#include "cinn/hlir/instruction/computation.h"

namespace cinn {
namespace hlir {
namespace instruction {

inline Computation::Builder* BinaryOpParamBuilderCheck(Instruction* a, Instruction* b) {
  auto* a_builder = InstructionGetComputationBuilder(a);
  auto* b_builder = InstructionGetComputationBuilder(b);
  CHECK(a_builder);
  CHECK_EQ(a_builder, b_builder) << "The computation builder of input parameters of a Binary instruction not match";
  return a_builder;
}

Instruction* Add(Instruction* a, Instruction* b) {
  auto* builder = BinaryOpParamBuilderCheck(a, b);
  return builder->AddInstruction(Instruction::CreateBinary(InstrCode::Add, a, b));
}
Instruction* Sub(Instruction* a, Instruction* b) {
  auto* builder = BinaryOpParamBuilderCheck(a, b);
  return builder->AddInstruction(Instruction::CreateBinary(InstrCode::Sub, a, b));
}
Instruction* Mul(Instruction* a, Instruction* b) {
  auto* builder = BinaryOpParamBuilderCheck(a, b);
  return builder->AddInstruction(Instruction::CreateBinary(InstrCode::Mul, a, b));
}
Instruction* Div(Instruction* a, Instruction* b) {
  auto* builder = BinaryOpParamBuilderCheck(a, b);
  return builder->AddInstruction(Instruction::CreateBinary(InstrCode::Div, a, b));
}

Instruction* Tanh(Instruction* x) {
  auto* builder = InstructionGetComputationBuilder(x);
  CHECK(builder);
  return builder->AddInstruction(Instruction::CreateUnary(InstrCode::Tanh, x));
}
Instruction* Abs(Instruction* x) {
  auto* builder = InstructionGetComputationBuilder(x);
  CHECK(builder);
  return builder->AddInstruction(Instruction::CreateUnary(InstrCode::Abs, x));
}
Instruction* Ceil(Instruction* x) {
  auto* builder = InstructionGetComputationBuilder(x);
  CHECK(builder);
  return builder->AddInstruction(Instruction::CreateUnary(InstrCode::Ceil, x));
}

Instruction* Dot(Instruction* a, Instruction* b) {
  auto* builder = BinaryOpParamBuilderCheck(a, b);
  return builder->AddInstruction(Instruction::CreateDot(a, b));
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

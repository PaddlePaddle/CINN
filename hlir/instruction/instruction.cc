#include "hlir/instruction/instruction.h"
#include "hlir/instruction/instructions.h"

namespace hlir {
namespace instruction {

void InstructionKind::SetFlag(unsigned int flag, bool x) {
  if (x) {
    (*reinterpret_cast<unsigned int *>(&kind_)) |= flag;
  } else {
    (*reinterpret_cast<unsigned int *>(&kind_)) &= ~flag;
  }
}

std::unique_ptr<Instruction> Instruction::CreateParameter(const Shape &shape, const std::string &name) {
  return std::unique_ptr<Instruction>(new ParameterInstruction(name, shape));
}

std::unique_ptr<Instruction> Instruction::CreateUnary(const Shape &shape, InstrCode instr_code, Instruction *arg0) {
  auto instr = std::unique_ptr<Instruction>(new Instruction(instr_code, shape));
  instr->AppendOperand(arg0);
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateBinary(const Shape &shape,
                                                       InstrCode instr_code,
                                                       Instruction *arg0,
                                                       Instruction *arg1) {
  auto instr = std::unique_ptr<Instruction>(new Instruction(instr_code, shape));
  instr->AppendOperand(arg0);
  instr->AppendOperand(arg1);
  return instr;
}

void Instruction::AppendOperand(Instruction *operand) { operands_.push_back(operand); }

std::unique_ptr<Instruction> Instruction::CreateCompare(const Shape &shape,
                                                        Instruction *arg0,
                                                        Instruction *arg1,
                                                        CompareDirection dire) {
  return std::unique_ptr<Instruction>(new CompareInstruction(shape, arg0, arg1, dire));
}

std::unique_ptr<Instruction> Instruction::CreateDot(const Shape &shape, Instruction *arg0, Instruction *arg1) {
  auto instr = std::unique_ptr<Instruction>(new Instruction(InstrCode ::Dot, shape));
  instr->AppendOperand(arg0);
  instr->AppendOperand(arg1);
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateReduce(const Shape &shape,
                                                       Instruction *operand,
                                                       Instruction *init_value,
                                                       const std::vector<int> &reduce_dimensions,
                                                       Computation *reduce_computation) {
  auto instr = std::unique_ptr<Instruction>(
      new ReduceInstruction(shape, operand, init_value, reduce_dimensions, reduce_computation));
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateBroadcast(const Shape &shape,
                                                          Instruction *arg0,
                                                          const std::vector<int> &dimensions) {
  auto instr = std::unique_ptr<Instruction>(new BroadcastInstruction(shape, dimensions));
  instr->AppendOperand(arg0);
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateTranspose(const Shape &shape,
                                                          Instruction *arg0,
                                                          const std::vector<int> &dimensions) {
  auto instr = std::unique_ptr<Instruction>(new TransposeInstruction(shape, dimensions));
  instr->AppendOperand(arg0);
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateCall(const Shape &shape,
                                                     const std::vector<Instruction *> &args,
                                                     Computation *computation) {
  auto instr = std::unique_ptr<Instruction>(new Instruction(InstrCode::Call, shape));
  for (auto *arg : args) instr->AppendOperand(arg);
  instr->called_computations_.push_back(computation);
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateNary(const Shape &shape,
                                                     const std::vector<Instruction *> &args,
                                                     InstrCode instr_code) {
  switch (instr_code) {
    case InstrCode::Add:
    case InstrCode::Sub:
    case InstrCode::Mul:
    case InstrCode::Div:
    case InstrCode::Abs:
    case InstrCode::And:
    case InstrCode::Or:
    case InstrCode::Not:
      break;

    default:
      NOT_IMPLEMENTED
  }

  auto instr = std::unique_ptr<Instruction>(new Instruction(instr_code, shape));
  for (auto *arg : args) instr->AppendOperand(arg);

  return instr;
}

const Instruction *Instruction::operand(int i) const {
  CHECK_LT(i, operands_.size());
  return operands_[i];
}

}  // namespace instruction
}  // namespace hlir

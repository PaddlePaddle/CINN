#include "hlir/instruction/instruction.h"
#include <sstream>
#include "cinn/common/macros.h"
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

InstructionKind &InstructionKind::set_elementwise(bool x) {
  SetFlag(KindAsInt(Kind::Elementwise), x);
  return *this;
}

std::unique_ptr<Instruction> Instruction::CreateParameter(int param_offset,
                                                          const Shape &shape,
                                                          const std::string &name,
                                                          const ParameterConfig &config) {
  auto res = std::unique_ptr<Instruction>(new ParameterInstruction(param_offset, name, shape));
  res->set_type(config.type);
  return res;
}

std::unique_ptr<Instruction> Instruction::CreateUnary(const Shape &shape, InstrCode instr_code, Instruction *arg0) {
  auto instr = std::unique_ptr<Instruction>(new Instruction(instr_code, shape));
  instr->AppendOperand(arg0);
  instr->set_type(arg0->type());
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateBinary(const Shape &shape,
                                                       InstrCode instr_code,
                                                       Instruction *arg0,
                                                       Instruction *arg1) {
  switch (instr_code) {
    case InstrCode::Add:
    case InstrCode::Mul:
    case InstrCode::Div:
    case InstrCode::Sub:
      break;
    default:
      LOG(FATAL) << "Not supported binary instruction type: " << InstrCodeToString(instr_code);
  }

  auto instr = std::unique_ptr<Instruction>(new Instruction(instr_code, shape));
  instr->AppendOperand(arg0);
  instr->AppendOperand(arg1);
  CHECK_EQ(arg0->type(), arg1->type());
  instr->set_type(arg0->type());
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
  CHECK_EQ(arg0->type(), arg1->type());
  instr->AppendOperand(arg0);
  instr->AppendOperand(arg1);
  instr->set_type(arg0->type());
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
  auto instr = std::unique_ptr<Instruction>(new CallInstruction(shape, computation, args));
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

std::unique_ptr<Instruction> Instruction::CreateConstant(const Shape &shape,
                                                         const std::vector<char> &buf,
                                                         const ConstantConfig &config) {
  auto instr = std::unique_ptr<Instruction>(new ConstantInstruction(shape, buf));
  instr->set_type(config.type);
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateCustomCall(const Shape &shape,
                                                           const std::vector<Instruction *> &args,
                                                           const std::string &target,
                                                           const std::string &tag) {
  return std::unique_ptr<Instruction>(new CustomCallInstruction(shape, args, target, tag));
}

const Instruction *Instruction::operand(int i) const {
  CHECK_LT(i, operands_.size());
  return operands_[i];
}

std::string Instruction::to_debug_string() {
  std::stringstream ss;

  ss << "%" << id() << " :" << type() << " " << shape().to_debug_string();
  ss << " = ";

  ss << InstrCodeToString(instr_code_) << "(";
  if (!operands_.empty()) {
    for (int i = 0; i < operands_.size() - 1; i++) {
      CHECK(operands_[i]);
      ss << "%" << operands_[i]->id();
      ss << ", ";
    }
    if (!operands_.empty()) {
      ss << "%" << operands_.back()->id();
    }
  }
  ss << ")";

  if (comment_) {
    ss << "\t;; " << comment();
  }
  return ss.str();
}

void Instruction::set_type(const type_t &type) {
  CHECK(type.valid());
  type_ = type;
}

void Instruction::AddUser(Instruction *user) {
  outlinks_.insert(user);
  user->inlinks_.insert(this);
}

void Instruction::RemoveUser(Instruction *user) {
  outlinks_.erase(user);
  user->inlinks_.erase(this);
}

}  // namespace instruction
}  // namespace hlir

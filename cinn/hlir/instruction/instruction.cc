#include "cinn/hlir/instruction/instruction.h"

#include <sstream>

#include "cinn/common/macros.h"
#include "cinn/hlir/instruction/computation.h"
#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/shape_inference.h"

namespace cinn {
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

std::unique_ptr<Instruction> Instruction::CreateUnary(InstrCode instr_code, Instruction *arg0, const Shape &shape) {
  auto instr = std::unique_ptr<Instruction>(new Instruction(instr_code, arg0->shape()));
  instr->AppendOperand(arg0);
  instr->set_type(arg0->type());
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateBinary(InstrCode instr_code,
                                                       Instruction *arg0,
                                                       Instruction *arg1,
                                                       const Shape &shape) {
  switch (instr_code) {
    case InstrCode::Add:
    case InstrCode::Mul:
    case InstrCode::Div:
    case InstrCode::Sub:
      break;
    default:
      LOG(FATAL) << "Not supported binary instruction type: " << InstrCodeToString(instr_code);
  }

  auto _shape = shape.empty() ? BinaryInferenceShape(arg0, arg1) : shape;

  auto instr = std::unique_ptr<Instruction>(new Instruction(instr_code, _shape));
  instr->AppendOperand(arg0);
  instr->AppendOperand(arg1);
  CHECK_EQ(arg0->type(), arg1->type());
  instr->set_type(arg0->type());
  return instr;
}

void Instruction::AppendOperand(Instruction *operand) { operands_.push_back(operand); }

std::unique_ptr<Instruction> Instruction::CreateCompare(Instruction *arg0,
                                                        Instruction *arg1,
                                                        CompareDirection dire,
                                                        const Shape &shape) {
  return std::unique_ptr<Instruction>(new CompareInstruction(shape, arg0, arg1, dire));
}

std::unique_ptr<Instruction> Instruction::CreateDot(Instruction *arg0, Instruction *arg1, const Shape &shape) {
  auto _shape = shape.empty() ? DotInferenceShape(arg0, arg1) : shape;

  auto instr = std::unique_ptr<Instruction>(new Instruction(InstrCode ::Dot, _shape));
  CHECK_EQ(arg0->type(), arg1->type());
  instr->AppendOperand(arg0);
  instr->AppendOperand(arg1);
  instr->set_type(arg0->type());
  return instr;
}

std::unique_ptr<Instruction> Instruction::CreateReduce(Instruction *operand,
                                                       Instruction *init_value,
                                                       const std::vector<int> &reduce_dimensions,
                                                       Computation *reduce_computation,
                                                       const Shape &shape) {
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

std::unique_ptr<Instruction> Instruction::CreateCall(const std::vector<Instruction *> &args,
                                                     const std::string &ret_name,
                                                     const Shape &shape,
                                                     const cinn::common::Type &type,
                                                     const Computation *computation) {
  auto instr = std::unique_ptr<Instruction>(new CallInstruction(computation, args, {shape}, {ret_name}, {type}));
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

std::unique_ptr<Instruction> Instruction::CreateTuple(const Instruction *call) {
  return std::unique_ptr<Instruction>(new Tuple(call));
}

std::unique_ptr<Instruction> Instruction::CreateTuple(const std::vector<const Instruction *> &items) {
  return std::unique_ptr<Instruction>(new Tuple(items));
}

std::unique_ptr<Instruction> Instruction::CreateTupleGet(const Instruction *tuple, int offset) {
  return std::unique_ptr<Instruction>(new TupleGet(tuple, offset));
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
}  // namespace cinn

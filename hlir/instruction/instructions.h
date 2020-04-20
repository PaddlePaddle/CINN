#pragma once
#include <string>

#include "hlir/instruction/instruction.h"

namespace hlir {
namespace instruction {

class ParameterInstruction : public Instruction {
 public:
  ParameterInstruction(const std::string& name, const Shape& shape)
      : name_(name), Instruction(InstrCode::Parameter, shape) {}

 private:
  std::string name_;
};

class CompareInstruction : public Instruction {
 public:
  CompareInstruction(const Shape& shape, Instruction* arg0, Instruction* arg1, CompareDirection direction)
      : Instruction(InstrCode::Compare, shape), direction_(direction) {
    AppendOperand(arg0);
    AppendOperand(arg1);
  }

 private:
  CompareDirection direction_;
};

class ReduceInstruction : public Instruction {
 public:
  ReduceInstruction(const Shape& shape,
                    Instruction* arg0,
                    Instruction* init_value,
                    const std::vector<int>& reduce_dimensions,
                    Computation* reduce_computation)
      : Instruction(InstrCode::Reduce, shape),
        init_value_(init_value),
        reduce_dimensions_(reduce_dimensions),
        reduce_computation_(reduce_computation) {
    AppendOperand(arg0);
    AppendOperand(init_value);
  }

 private:
  Instruction* init_value_{};
  std::vector<int> reduce_dimensions_;
  Computation* reduce_computation_{};
};

class BroadcastInstruction : public Instruction {
 public:
  BroadcastInstruction(const Shape& shape, const std::vector<int>& dimensions)
      : Instruction(InstrCode::Broadcast, shape), dimensions_(dimensions) {}

 private:
  std::vector<int> dimensions_;
};

class TransposeInstruction : public Instruction {
 public:
  TransposeInstruction(const Shape& shape, const std::vector<int>& dimensions)
      : Instruction(InstrCode::Transpose, shape), dimensions_(dimensions) {}

 private:
  std::vector<int> dimensions_;
};

}  // namespace instruction
}  // namespace hlir

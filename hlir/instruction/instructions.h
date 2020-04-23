#pragma once
#include <string>

#include "hlir/instruction/instruction.h"

namespace hlir {
namespace instruction {

class ParameterInstruction : public Instruction {
 public:
  ParameterInstruction(int param_offset, const std::string& name, const Shape& shape)
      : name_(name), Instruction(InstrCode::Parameter, shape), param_offset_(param_offset) {}

  std::string to_debug_string() override;

  const std::string& name() const { return name_; }

  std::string id() const override;

  int param_offset() const { return param_offset_; }

 private:
  std::string name_;
  int param_offset_{-1};
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

class ConstantInstruction : public Instruction {
 public:
  ConstantInstruction(const Shape& shape, const std::vector<char>& data)
      : Instruction(InstrCode::Constant, shape), data_(data) {}

 private:
  std::vector<char> data_;
};

class CallInstruction : public Instruction {
 public:
  CallInstruction(const Shape& shape, Computation* computation, const std::vector<Instruction*>& args)
      : Instruction(InstrCode::Call, shape), computation_(computation) {
    for (auto* arg : args) {
      AppendOperand(arg);
    }
  }

  std::string to_debug_string() override;

 private:
  Computation* computation_{};
};

class CustomCallInstruction : public Instruction {
 public:
  CustomCallInstruction(const Shape& shape,
                        const std::vector<Instruction*>& args,
                        const std::string& call_target,
                        const std::string& tag)
      : Instruction(InstrCode::CustomCall, shape), call_target_(call_target), args_(args) {}

 private:
  std::string call_target_;
  std::vector<Instruction*> args_;
};

}  // namespace instruction
}  // namespace hlir

#pragma once

#include <string>
#include <vector>

#include "cinn/common/type.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/op.h"

namespace cinn {
namespace frontend {

class BaseBuilder {
 public:
  explicit BaseBuilder(const std::string& name);

  Program Build();

  Placeholder CreateInput(const common::Type& type, const std::vector<int>& shape, const std::string& id_hint = "");

  // name of this builder
  const std::string& name() { return name_; }

  virtual ~BaseBuilder() {}

 protected:
  void AppendInstruction(const Instruction& instr) { instrs_.push_back(instr); }

  void InferShape(Instruction instr) const;

  std::string name_;
  std::vector<Instruction> instrs_;
  std::vector<Variable> inputs_;
};

}  // namespace frontend
}  // namespace cinn

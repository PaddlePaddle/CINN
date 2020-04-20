#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hlir/instruction/instruction.h"

namespace hlir {
namespace instruction {

struct Module;
/**
 * A computation is analogous to a function. It has some inputs(parameters) and returns exactly one value (the value of
 * its root node).
 */
class Computation {
 public:
  class Builder {
   public:
    explicit Builder(const std::string& name) : name_(name) {}

    //! Build and return a Computation.
    std::unique_ptr<Computation> Build();

    Instruction* AddInstruction(std::unique_ptr<Instruction> instruction) {
      instructions_.push_back(std::move(instruction));
      last_added_instruction_ = instructions_.back().get();
      return last_added_instruction_;
    }

   private:
    std::string name_;
    std::vector<std::unique_ptr<Instruction>> instructions_;
    Instruction* last_added_instruction_{};
  };

  //! Add an instruction to the computation.
  Instruction* AddInstruction(std::unique_ptr<Instruction> instruction);

  //! Remove an instruction from the computation. The instruction must have no users, the instruction will be
  //! deallocated.
  bool RemoveInstruction(Instruction* instruction);

 private:
  //! Module containing this computation.
  Module* module_{};
};

}  // namespace instruction
}  // namespace hlir

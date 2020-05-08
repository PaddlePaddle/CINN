#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hlir/instruction/context.h"
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
    Builder(Context* context, const std::string& name) : context_(*context), name_(context->new_computation_id(name)) {}

    /*
     * Build and return a Computation.
     */
    std::unique_ptr<Computation> Build();

    /**
     * Add an instruction to th computation.
     * @param instruction The instruction to add.
     * @param comment The optional comment of this instruction.
     * @return The added instruction.
     */
    Instruction* AddInstruction(std::unique_ptr<Instruction>&& instruction, const std::string& comment = "");

    /**
     * Shape of the return value.
     */
    inline const Shape& shape() const;

    /**
     * Type of the return value.
     */
    inline const type_t& type() const;

   private:
    Context& context_;
    std::string name_;
    bool is_built_{false};
    std::vector<std::unique_ptr<Instruction>> instructions_;
    Instruction* last_added_instruction_{};
  };

  //! Get the parameters of the computation(all are tensors).
  std::vector<Instruction*> GetParameters() const;
  //! Get the constant instructions, those are also tensors.
  std::vector<Instruction*> GetConstants() const;
  //! Get the intermediate instructions, thery are not parameter, constant or root.
  std::vector<Instruction*> GetIntermediates() const;
  //! Get the variables (usually represents the dynamic dimension in shape).
  std::vector<cinn::Var> CollecVars() const;

  const std::vector<std::unique_ptr<Instruction>>& instructions() const { return instructions_; }

  const Instruction* root_instruction() const {
    CHECK(!instructions_.empty());
    return instructions_.back().get();
  }

  //! Remove an instruction from the computation. The instruction must have no users, the instruction will be
  //! deallocated.
  bool RemoveInstruction(Instruction* instruction);

  std::string to_debug_string() const;

  const std::string& name() const { return name_; }

 private:
  Computation(std::vector<std::unique_ptr<Instruction>>&& instructions, const std::string& name)
      : instructions_(std::move(instructions)), name_(name) {}

  std::vector<std::unique_ptr<Instruction>> instructions_;

  std::string name_;
  //! Module containing this computation.
  Module* module_{};
};

}  // namespace instruction
}  // namespace hlir

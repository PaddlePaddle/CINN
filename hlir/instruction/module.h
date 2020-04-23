#pragma once
#include "hlir/instruction/computation.h"
#include "hlir/instruction/instruction.h"

namespace hlir {
namespace instruction {

class Module {
 public:
  Module(const std::string& name) : name_(name) {}
  Module(std::vector<std::unique_ptr<Computation>>&& computations, const std::string& name);

  /**
   * Add a computation to the module.
   */
  Computation* AddComputation(std::unique_ptr<Computation>&& computation);

  /**
   * Add a computation as the entry.
   */
  Computation* AddEntryComputation(std::unique_ptr<Computation>&& computation);

  const std::string& name() const { return name_; }

  const std::map<std::string, std::unique_ptr<Computation>>& computations() const { return computations_; }

  std::string to_debug_string() const;

 private:
  std::map<std::string, std::unique_ptr<Computation>> computations_;

  Computation* entry_computation_{};

  std::string name_;
};

}  // namespace instruction
}  // namespace hlir

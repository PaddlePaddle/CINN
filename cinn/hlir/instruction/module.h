#pragma once
#include <map>
#include <string>
#include <vector>

#include "cinn/hlir/instruction/computation.h"
#include "cinn/hlir/instruction/instruction.h"

namespace cinn {
namespace hlir {
namespace instruction {

class Module {
 public:
  explicit Module(const std::string& name) : name_(name) {}
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

  /**
   * Look up a computation in the module, returns null if not exist.
   * @param name Name of the target computation.
   * @return pointer of the computation if exists or null.
   */
  const Computation* LookupComputation(const std::string& name) const;

  const std::map<std::string, std::unique_ptr<Computation>>& computations() const { return computations_; }

  const Computation* entry_computation() const;

  std::string to_debug_string() const;

 private:
  std::map<std::string, std::unique_ptr<Computation>> computations_;

  Computation* entry_computation_{};

  std::string name_;
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

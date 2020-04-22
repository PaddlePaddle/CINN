#include "hlir/instruction/module.h"

namespace hlir {
namespace instruction {

Computation *Module::AddEntryComputation(std::unique_ptr<Computation> &&computation) {
  auto name           = computation->name();
  computations_[name] = std::move(computation);
  entry_computation_  = computations_[name].get();
  return entry_computation_;
}

Computation *Module::AddComputation(std::unique_ptr<Computation> &&computation) {
  auto name = computation->name();
  CHECK(!computations_.count(name)) << "Get module with the duplicate name " << computation->name();
  computations_[name] = std::move(computation);
  return computations_[name].get();
}

Module::Module(std::vector<std::unique_ptr<Computation>> &&computations, const std::string &name) : name_(name) {
  for (auto &computation : computations) {
    AddComputation(std::move(computation));
  }
}

std::string Module::to_debug_string() const {
  std::stringstream ss;

  ss << "{\n";
  ss << ";; == module == " << name_ << "\n";
  ss << "\n";

  for (auto &comp : computations_) {
    ss << comp.second->to_debug_string();
  }

  ss << "\n}\n";

  return ss.str();
}

}  // namespace instruction
}  // namespace hlir
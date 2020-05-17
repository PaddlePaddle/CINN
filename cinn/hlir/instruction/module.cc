#include "cinn/hlir/instruction/module.h"

namespace cinn {
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

const Computation *Module::entry_computation() const {
  // CHECK(entry_computation_) << "No entry computation is set";
  return entry_computation_;
}

const Computation *Module::LookupComputation(const std::string &name) const {
  auto it = computations_.find(name);
  if (it == computations_.end()) return nullptr;
  return it->second.get();
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

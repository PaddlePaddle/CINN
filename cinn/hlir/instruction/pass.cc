#include "cinn/hlir/instruction/pass.h"

#include "cinn/hlir/instruction/module_group.h"

namespace cinn {
namespace hlir {
namespace instruction {

bool ModulePass::RunOnModuleGroup(ModuleGroup* module_group) {
  bool changed = false;
  for (auto* module : *module_group) {
    changed |= Run(module);
  }
  return changed;
}

void PassRegistry::Insert(const std::string& name, PassRegistry::creator_t&& creator) {
  CHECK(!data_.count(name)) << "Duplicate register pass [" << name << "]";
  data_[name] = creator;
}

bool PassRegistry::Has(const std::string& name) const { return data_.count(name); }

std::unique_ptr<PassInterface> PassRegistry::Create(const std::string& name) {
  auto it = data_.find(name);
  if (it == data_.end()) return nullptr;
  return it->second();
}

std::unique_ptr<PassInterface> PassRegistry::CreatePromised(const std::string& name) {
  auto x = Create(name);
  CHECK(x) << "No pass creator found for [" << name << "]";
  return x;
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
